#!/usr/bin/env python3
"""
router.py
=========
Semantic traffic router for the 10-K comparison RAG system
(Alphabet, Amazon, Microsoft).

Classifies an incoming financial query into one of three computational tiers
using a rapid, deterministic Gemini classification call (google-genai SDK,
gemini-2.5-flash, temperature=0.0, JSON-enforced output):

    Tier 1  "gemini"  Cloud / Advanced   k=12  cross-company, multi-hop,
                                               strategic / "why" analysis
    Tier 2  "qwen"    Local / Fast       k=4   single-company KPI / number /
                                               date lookups
    Tier 3  "ollama"  Local / Narrative  k=4   single-entity textual notes,
                                               disclosures, accounting text

Public API (import from your query engine):
    get_router_decision(question: str) -> tuple[str, str, int]
        returns (route, reason, recommended_k)

Resilience: any API timeout, network exception, or validation failure falls
back to ("gemini", "Routing bypass due to exception catch", 12) so the
pipeline never stalls on the router.

Environment:
    GOOGLE_API_KEY  required (google-genai reads it automatically)

Dependencies:
    pip install google-genai
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache

from google import genai
from google.genai import types
from google.genai.errors import APIError

logger = logging.getLogger("router")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROUTER_MODEL = "gemini-2.5-flash"
REQUEST_TIMEOUT_MS = 15_000  # 15s hard cap on the classification call

VALID_ROUTES = {"gemini", "qwen", "ollama"}
ROUTE_K = {"gemini": 12, "qwen": 4, "ollama": 4}

FALLBACK_DECISION: tuple[str, str, int] = (
    "gemini",
    "Routing bypass due to exception catch",
    12,
)

# JSON schema enforced on the model output (google-genai structured output)
RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "route": types.Schema(
            type=types.Type.STRING,
            enum=["gemini", "qwen", "ollama"],
        ),
        "reason": types.Schema(type=types.Type.STRING),
        "recommended_k": types.Schema(
            type=types.Type.INTEGER,
            enum=["12", "4"],
        ),
    },
    required=["route", "reason", "recommended_k"],
)

SYSTEM_INSTRUCTION = """You are a deterministic query router for a financial
RAG system over SEC 10-K filings (Alphabet, Amazon, Microsoft). Classify each
incoming question into exactly one computational tier. Respond ONLY with a
raw JSON object, no markdown fences, no commentary.

TIER DEFINITIONS AND TRIGGERS:

1. "gemini" (Cloud / Advanced, recommended_k=12)
   High-complexity cognitive load. Route here when the query involves:
   - Cross-company comparisons (two or more of Alphabet/Amazon/Microsoft)
   - Multi-hop reasoning (answer requires combining several facts)
   - Macroeconomic synthesis or trend analysis across years
   - Qualitative strategic analysis: "why", "strategy", "impact",
     "outlook", "how does X affect Y"
   - Any query spanning multiple corporate entities

2. "qwen" (Local / Fast, recommended_k=4)
   Low-complexity lookup. Route here when the query is:
   - A single-company KPI extraction (revenue, net income, EPS, margins)
   - A direct quantitative lookup or balance sheet line item
   - A request for a specific number, figure, or specific date
   - Answerable from one or two chunks of one filing

3. "ollama" (Local / Narrative, recommended_k=4)
   Secondary local pathway for text-heavy single-entity reads:
   - Reading block textual notes or footnotes
   - Disclosures (risk factors, legal proceedings, controls) of ONE company
   - Qualitative accounting statements or policy descriptions of ONE company

EXAMPLES:

Q: "Compare the R&D spending of Microsoft and Alphabet in 2025."
A: {"route": "gemini", "reason": "Cross-company quantitative comparison spanning two entities", "recommended_k": 12}

Q: "Why did Amazon's operating margin improve, and how does its strategy differ from Microsoft's?"
A: {"route": "gemini", "reason": "Qualitative strategic 'why' analysis with multi-hop cross-company reasoning", "recommended_k": 12}

Q: "How did macro conditions affect cloud revenue growth across all three companies?"
A: {"route": "gemini", "reason": "Macroeconomic synthesis spanning multiple corporate entities", "recommended_k": 12}

Q: "What was Microsoft's total revenue for fiscal year 2025?"
A: {"route": "qwen", "reason": "Single-company direct KPI lookup", "recommended_k": 4}

Q: "What is Amazon's diluted EPS for 2024?"
A: {"route": "qwen", "reason": "Specific number extraction from one filing", "recommended_k": 4}

Q: "On what date did Alphabet's fiscal year 2023 end?"
A: {"route": "qwen", "reason": "Specific date lookup for a single company", "recommended_k": 4}

Q: "Summarize the cybersecurity risk disclosures in Microsoft's Item 1A."
A: {"route": "ollama", "reason": "Text-heavy single-entity disclosure read", "recommended_k": 4}

Q: "What does Amazon's footnote on revenue recognition policy say?"
A: {"route": "ollama", "reason": "Block textual accounting note from one company", "recommended_k": 4}

Q: "Describe Alphabet's legal proceedings section."
A: {"route": "ollama", "reason": "Qualitative narrative disclosure for a single entity", "recommended_k": 4}

Output strictly: {"route": "...", "reason": "...", "recommended_k": 12 or 4}
"""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    """Lazily construct (and cache) the google-genai client."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_decision(payload: dict) -> tuple[str, str, int]:
    """
    Strictly validate the model's JSON payload.
    Raises ValueError on any deviation from the contract.
    """
    if not isinstance(payload, dict):
        raise ValueError("Router payload is not a JSON object.")

    route = payload.get("route")
    reason = payload.get("reason")
    k = payload.get("recommended_k")

    if route not in VALID_ROUTES:
        raise ValueError(f"Invalid route value: {route!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Missing or empty 'reason' field.")

    # Coerce k defensively (models occasionally emit numbers as strings),
    # then enforce the per-route cap regardless of what the model said.
    try:
        k = int(k)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid recommended_k value: {k!r}")

    expected_k = ROUTE_K[route]
    if k != expected_k:
        logger.warning(
            "Model suggested k=%d for route %s; capping to %d.",
            k, route, expected_k,
        )
        k = expected_k

    return route, reason.strip(), k


def _strip_markdown_fences(text: str) -> str:
    """
    Defensive cleanup: remove ```json ... ``` fences if the model added them
    despite JSON enforcement, so json.loads always receives a raw object.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1] if cleaned.count("```") >= 2 else cleaned
        cleaned = cleaned.lstrip("json").strip()
    return cleaned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_router_decision(question: str) -> tuple[str, str, int]:
    """
    Classify a financial query into a computational tier.

    Args:
        question: The raw user query.

    Returns:
        (route, reason, recommended_k) where route is one of
        "gemini" | "qwen" | "ollama" and recommended_k is 12 or 4.

    Never raises: any exception (timeout, network, validation) resolves to
    the resilient fallback ("gemini", "Routing bypass due to exception
    catch", 12).
    """
    if not question or not question.strip():
        return FALLBACK_DECISION

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=ROUTER_MODEL,
            contents=question.strip(),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                # Router calls should be instant; disable thinking budget.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

        raw = _strip_markdown_fences(response.text or "")
        payload = json.loads(raw)
        route, reason, k = _validate_decision(payload)

        logger.info("Routed to %-6s (k=%-2d): %s", route, k, reason)
        return route, reason, k

    except (APIError, TimeoutError, ConnectionError) as exc:
        logger.error("Router API/network failure: %s", exc)
        return FALLBACK_DECISION
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Router output validation failure: %s", exc)
        return FALLBACK_DECISION
    except Exception as exc:  # absolute safety net: router must never crash
        logger.error("Unexpected router failure: %s", exc)
        return FALLBACK_DECISION
