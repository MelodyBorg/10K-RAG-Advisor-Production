#!/usr/bin/env python3
"""
router.py
=========
Semantic traffic router for the 10-K comparison RAG system
(Alphabet, Amazon, Microsoft).

Classifies an incoming financial query into one of four computational tiers
using a rapid, deterministic OpenAI classification call (gpt-4o-mini,
temperature=0.0, strict JSON-schema-enforced output). Routing runs on
gpt-4o-mini deliberately: it is cheap, fast, and — critically — does NOT
consume the Gemini free-tier quota that the main Gemini answer tier depends
on (routing on Gemini caused every routing call to compete with actual
answers for the same 5/min quota, collapsing Auto-Pilot to permanent
fallback).

    Tier 1  "gemini"  Cloud / Advanced   k=12  cross-company, multi-hop,
                                               multi-entity synthesis
    Tier 2  "gpt5"    Cloud / Advanced   k=12  single-company qualitative /
                                               strategic "why" reasoning
                                               (GPT-5.4, strongest reasoner)
    Tier 3  "qwen"    Local / Fast       k=4   single-company KPI / number /
                                               date lookups
    Tier 4  "ollama"  Local / Narrative  k=4   single-entity textual notes,
                                               disclosures, accounting text

Public API (import from your query engine):
    get_router_decision(question: str) -> tuple[str, str, int]
        returns (route, reason, recommended_k)

Resilience: any API timeout, network exception, or validation failure falls
back to ("gemini", <reason explaining the failure>, 12) so the pipeline
never stalls on the router. Rate-limit failures carry a distinct reason so
the UI can tell the user the router was skipped because a limit was hit.

Environment:
    OPENAI_API_KEY  required (the openai SDK reads it automatically)

Dependencies:
    pip install openai
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

logger = logging.getLogger("router")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-4o-mini")
REQUEST_TIMEOUT_S = 15.0  # 15s hard cap on the classification call

VALID_ROUTES = {"gemini", "gpt5", "qwen", "ollama"}
ROUTE_K = {"gemini": 12, "gpt5": 12, "qwen": 4, "ollama": 4}

# Generic fallback for unexpected failures. Rate-limit failures get their own
# reason (below) so the UI can surface WHY routing was skipped.
FALLBACK_DECISION: tuple[str, str, int] = (
    "gemini",
    "Router bypassed (unexpected error contacting the routing model) — "
    "defaulted to Gemini.",
    12,
)

RATE_LIMIT_FALLBACK: tuple[str, str, int] = (
    "gemini",
    "Router bypassed — the routing model's API rate limit or quota was "
    "reached, so this query was not classified and defaulted to Gemini.",
    12,
)

MISSING_KEY_FALLBACK: tuple[str, str, int] = (
    "gemini",
    "Router bypassed — OPENAI_API_KEY is not configured, so semantic "
    "routing is unavailable and the query defaulted to Gemini.",
    12,
)

# Strict JSON schema enforced on the model output (OpenAI structured output)
RESPONSE_JSON_SCHEMA = {
    "name": "route_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "route": {
                "type": "string",
                "enum": ["gemini", "gpt5", "qwen", "ollama"],
            },
            "reason": {"type": "string"},
            "recommended_k": {"type": "integer", "enum": [12, 4]},
        },
        "required": ["route", "reason", "recommended_k"],
        "additionalProperties": False,
    },
}

SYSTEM_INSTRUCTION = """You are a deterministic query router for a financial
RAG system over SEC 10-K filings (Alphabet, Amazon, Microsoft). Classify each
incoming question into exactly one computational tier. Respond ONLY with a
raw JSON object, no markdown fences, no commentary.

TIER DEFINITIONS AND TRIGGERS:

1. "gemini" (Cloud / Advanced, recommended_k=12)
   High-complexity CROSS-COMPANY cognitive load. Route here when the query
   spans TWO OR MORE of Alphabet/Amazon/Microsoft:
   - Cross-company comparisons (two or more of the three entities)
   - Multi-hop reasoning that combines facts from several filings
   - Macroeconomic synthesis or trend analysis across companies/years
   - Any query spanning multiple corporate entities

2. "gpt5" (Cloud / Advanced, GPT-5.4, recommended_k=12)
   High-complexity SINGLE-COMPANY reasoning that needs strong qualitative
   analysis (not just a lookup, and not cross-company). Route here when the
   query targets ONE company and asks for:
   - Strategic / causal analysis: "why", "strategy", "impact", "outlook",
     "how does X affect Y" for a single entity
   - Interpretation or synthesis of that company's narrative (implications of
     its risk factors, what its MD&A means, drivers behind a metric)
   - Open-ended analytical or advisory questions about one company that go
     beyond extracting a stated number or reading text verbatim

3. "qwen" (Local / Fast, recommended_k=4)
   Low-complexity lookup. Route here when the query is:
   - A single-company KPI extraction (revenue, net income, EPS, margins)
   - A direct quantitative lookup or balance sheet line item
   - A request for a specific number, figure, or specific date
   - Answerable from one or two chunks of one filing

4. "ollama" (Local / Narrative, recommended_k=4)
   Secondary local pathway for text-heavy single-entity reads that need
   little reasoning (just surface the text):
   - Reading block textual notes or footnotes
   - Disclosures (risk factors, legal proceedings, controls) of ONE company
   - Qualitative accounting statements or policy descriptions of ONE company

DISAMBIGUATION:
   - Multiple companies mentioned            -> "gemini"
   - One company + interpret/why/analyze     -> "gpt5"
   - One company + extract a number/date     -> "qwen"
   - One company + read/summarize text as-is -> "ollama"

EXAMPLES:

Q: "Compare the R&D spending of Microsoft and Alphabet in 2025."
A: {"route": "gemini", "reason": "Cross-company quantitative comparison spanning two entities", "recommended_k": 12}

Q: "Why did Amazon's operating margin improve, and how does its strategy differ from Microsoft's?"
A: {"route": "gemini", "reason": "Multi-hop cross-company reasoning across two entities", "recommended_k": 12}

Q: "How did macro conditions affect cloud revenue growth across all three companies?"
A: {"route": "gemini", "reason": "Macroeconomic synthesis spanning multiple corporate entities", "recommended_k": 12}

Q: "Why did Amazon's liquidity change in 2025, and what does it signal about its strategy?"
A: {"route": "gpt5", "reason": "Single-company causal/strategic reasoning requiring qualitative analysis", "recommended_k": 12}

Q: "What are the main drivers behind Microsoft's cloud growth and its outlook?"
A: {"route": "gpt5", "reason": "Single-company interpretive analysis of drivers and outlook", "recommended_k": 12}

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
def _get_client() -> OpenAI:
    """Lazily construct (and cache) the OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_S)


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

    Never raises: any exception (timeout, network, validation, rate limit)
    resolves to a resilient Gemini fallback whose reason string explains
    what went wrong, so the UI can surface it to the user.
    """
    if not question or not question.strip():
        return FALLBACK_DECISION

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=ROUTER_MODEL,
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": RESPONSE_JSON_SCHEMA,
            },
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": question.strip()},
            ],
        )

        raw = _strip_markdown_fences(response.choices[0].message.content or "")
        payload = json.loads(raw)
        route, reason, k = _validate_decision(payload)

        logger.info("Routed to %-6s (k=%-2d): %s", route, k, reason)
        return route, reason, k

    except RateLimitError as exc:
        # Distinct fallback so the user learns a LIMIT (not a bug) was hit.
        logger.error("Router rate limit/quota failure: %s", exc)
        return RATE_LIMIT_FALLBACK
    except RuntimeError as exc:
        # Raised by _get_client when the API key is absent.
        logger.error("Router configuration failure: %s", exc)
        return MISSING_KEY_FALLBACK
    except (APIError, APITimeoutError, APIConnectionError,
            TimeoutError, ConnectionError) as exc:
        logger.error("Router API/network failure: %s", exc)
        return FALLBACK_DECISION
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Router output validation failure: %s", exc)
        return FALLBACK_DECISION
    except Exception as exc:  # absolute safety net: router must never crash
        logger.error("Unexpected router failure: %s", exc)
        return FALLBACK_DECISION
