#!/usr/bin/env python3
"""
app.py
======
Premium Streamlit front-end for the 10-K RAG Advisor.
Redesigned to mimic high-end fintech aesthetics (minimalist, floating cards,
sharp typography).

Cloud-deployment fixes:
    1. Safe st.secrets access (no crash when secrets.toml is absent)
    2. BACKEND_ERR detail is now DISPLAYED, not swallowed
    3. Local Ollama tiers are hidden automatically on Streamlit Cloud
       (set CLOUD_ONLY=0 locally to re-enable them)
"""

from __future__ import annotations

import logging
import os

import streamlit as st

logger = logging.getLogger("app")

# --- Backend Integration Guard & Infrastructure ---
BACKEND_OK = True
BACKEND_ERR = ""

try:
    # 1. Infrastructure Check: Load API Key (env first, secrets fallback).
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            api_key = None

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        raise RuntimeError(
            "GOOGLE_API_KEY not found. Please set it in your Space 'Variables' "
            "or 'Secrets' settings."
        )

    # OpenAI (GPT) key is OPTIONAL: only the GPT tier needs it. Load it from
    # secrets first, then fall back to the environment. We do NOT raise if it
    # is missing so the app still runs on the Gemini/local tiers; the GPT tier
    # surfaces a clear error only if the user actually selects it.
    _openai_key = None
    try:
        _openai_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        _openai_key = None
    if _openai_key:
        os.environ["OPENAI_API_KEY"] = _openai_key

    # 2. Module Import (fails here if a package is missing from requirements.txt)
    from query import run_ui_query
    from router import get_router_decision

except Exception as exc:
    BACKEND_OK = False
    BACKEND_ERR = f"{type(exc).__name__}: {exc}"


# --- Page Optimization Settings ---
st.set_page_config(page_title="10-K RAG Advisor", layout="wide",
                   initial_sidebar_state="collapsed")

# --- Cloud environment detection ---
# Streamlit Community Cloud has no local Ollama daemon, so local tiers are
# hidden there. Locally, everything is available (or set CLOUD_ONLY=1 to test).
IS_CLOUD = os.getenv("CLOUD_ONLY", "").strip() == "1" or \
    os.path.exists("/mount/src")  # Community Cloud repo mount path

# --- Custom Premium FinTech Styles (The Synex Aesthetic) ---
SYNEX_THEME_CSS = """
<style>
/* Global Canvas Surface: Sand & Minimalist Off-White */
.stApp {
    background-color: #F8F8F6;
    color: #111111;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Hide Default Streamlit Chrome Padding */
header {visibility: hidden;}
footer {visibility: hidden;}
.block-container { padding-top: 2rem !important; }

/* Brutalist Financial Headers */
.header-title {
    font-size: 3.8rem;
    font-weight: 500;
    letter-spacing: -0.04em;
    margin-bottom: 4px;
    line-height: 1.05;
    color: #111111;
}
.header-sub {
    font-size: 1.15rem;
    color: #555555;
    margin-bottom: 2.5rem;
    letter-spacing: -0.01em;
}

/* Rounded Pill Infrastructure Navigation Control Layout */
div.row-widget.stRadio > div {
    flex-direction: row;
    gap: 1.8rem;
    background: #FFFFFF;
    padding: 12px 28px;
    border-radius: 50px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.03);
    display: inline-flex;
    border: 1px solid #EBEBE9;
}
div.row-widget.stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 0.9rem;
    font-weight: 500;
    color: #555555;
}

/* User Prompt Presentation Area */
.query-card {
    background: transparent;
    border-left: 2px solid #111111;
    padding: 8px 0px 8px 24px;
    margin: 24px 0px;
    font-size: 1.25rem;
    font-weight: 400;
    color: #222222;
    letter-spacing: -0.02em;
}

/* System Response Block (Premium Charcoal Floating Container) */
.insight-card {
    background: #111210;
    color: #F8F8F6;
    border-radius: 20px;
    padding: 36px;
    margin: 20px 0px;
    box-shadow: 0px 12px 40px rgba(0,0,0,0.12);
    font-size: 1.05rem;
    line-height: 1.65;
}
.insight-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #777777;
    margin-bottom: 18px;
    border-bottom: 1px solid #252624;
    padding-bottom: 10px;
    font-weight: 600;
}

/* Observability Analytics Dropdown Layout */
[data-testid="stExpander"] {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    margin-bottom: 30px;
}
[data-testid="stExpander"] summary {
    color: #777777;
    font-weight: 500;
    font-size: 0.88rem;
    padding-left: 0px !important;
}
[data-testid="stExpander"] summary:hover {
    color: #111111;
}

/* Clean Document Cards Inside Expander Container */
.chunk-card {
    background: #FFFFFF;
    border: 1px solid #EAEAEA;
    border-radius: 12px;
    padding: 18px;
    margin-top: 12px;
    font-size: 0.88rem;
    color: #444444;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.01);
}
.chunk-meta {
    font-weight: 600;
    color: #111111;
    margin-bottom: 6px;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
}

/* UI System Component Form Fields Overrides */
.stChatInput {
    background-color: transparent !important;
}
.stChatInput textarea {
    background-color: #FFFFFF !important;
    color: #111111 !important;
    border: 1px solid #E2E2E0 !important;
    border-radius: 30px !important;
    padding: 14px 24px !important;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.02) !important;
}
</style>
"""
st.markdown(SYNEX_THEME_CSS, unsafe_allow_html=True)

# --- Conversational State Cache Manager ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Custom Screen Layout Functions ---
def display_dashboard_header():
    st.markdown('<div class="header-title">Clarity and control for<br>your portfolio insights.</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">Institutional filing intelligence across Alphabet, Amazon, and Microsoft.</div>', unsafe_allow_html=True)

def render_user_turn(query_text: str):
    st.markdown(f'<div class="query-card">Query: {query_text}</div>', unsafe_allow_html=True)

def render_system_turn(response_text: str):
    st.markdown(f"""
        <div class="insight-card">
            <div class="insight-label">System Insight</div>
            {response_text}
        </div>
    """, unsafe_allow_html=True)

def render_data_provenance(metadata_payload: dict):
    with st.expander("View Data Provenance & Infrastructure Routing"):
        st.write(f"**Execution Engine:** {str(metadata_payload.get('mode')).upper()} | **Top-K Retrieval Depth:** {metadata_payload.get('k', '?')}")
        if metadata_payload.get("router_reason"):
            st.write(f"**Routing System Logic:** {metadata_payload['router_reason']}")

        retrieved_documents = metadata_payload.get("docs") or []
        for index, doc in enumerate(retrieved_documents, start=1):
            doc_meta = getattr(doc, "metadata", {})
            doc_content = getattr(doc, "page_content", str(doc))[:600] + "..."
            st.markdown(f"""
                <div class="chunk-card">
                    <div class="chunk-meta">Source Document {index} — {doc_meta.get('company', 'Corporate Entity')} | {doc_meta.get('section', 'Filing Section')} | Page {doc_meta.get('page', '?')}</div>
                    <div>{doc_content}</div>
                </div>
            """, unsafe_allow_html=True)

# --- Render Initial Core Elements ---
display_dashboard_header()

if not BACKEND_OK:
    st.error("Infrastructure Disconnected. Please verify that package bindings are active.")
    # FIX: show the actual root cause so cloud failures are debuggable.
    st.code(BACKEND_ERR, language="text")
    st.caption(
        "Common causes: a package missing from requirements.txt "
        "(e.g. langchain-community), GOOGLE_API_KEY not set in App "
        "settings -> Secrets, or FAISS index folders not committed to the repo."
    )
    st.stop()

# Curved Horizontal Pill Navigation Row
st.write("")
TIER_OPTIONS = [
    "Auto-Pilot (Semantic Routing)",
    "Gemini 2.5 Flash (Cloud)",
    "GPT-4o (OpenAI Cloud)",
    "GPT-5.4 (OpenAI Cloud)",
]
if not IS_CLOUD:
    TIER_OPTIONS += ["Qwen3 (Local KPI)", "Mistral (Local Narrative)"]

selected_infrastructure = st.radio(
    "Active Computational Tier Selector",
    TIER_OPTIONS,
    horizontal=True,
    label_visibility="hidden"
)

if IS_CLOUD:
    st.caption("Running on Streamlit Cloud — local Ollama tiers (Qwen3 / Mistral) "
               "are unavailable here; Auto-Pilot resolves everything to Gemini.")

# Operational Mapping Configuration Tuple
INFRASTRUCTURE_REGISTRY = {
    "Auto-Pilot (Semantic Routing)": ("auto", None),
    "Gemini 2.5 Flash (Cloud)": ("gemini", 12),
    "GPT-4o (OpenAI Cloud)": ("gpt4o", 12),
    "GPT-5.4 (OpenAI Cloud)": ("gpt5", 12),
    "Qwen3 (Local KPI)": ("qwen", 4),
    "Mistral (Local Narrative)": ("ollama", 4)
}

# --- Conversational History Loop Re-render ---
for message in st.session_state.messages:
    if message["role"] == "user":
        render_user_turn(message["content"])
    else:
        render_system_turn(message["content"])
        if message.get("meta"):
            render_data_provenance(message["meta"])

# --- Chat Input Capture Component ---
user_question = st.chat_input("Query corporate filings...")

if user_question:
    # Append input values to history thread
    st.session_state.messages.append({"role": "user", "content": user_question, "meta": None})
    render_user_turn(user_question)

    execution_mode_key, effective_k_value = INFRASTRUCTURE_REGISTRY[selected_infrastructure]
    routing_system_rationale = None

    with st.spinner("Synthesizing parameters..."):
        # Engage router if in autopilot configuration state
        if execution_mode_key == "auto":
            try:
                execution_mode_key, routing_system_rationale, effective_k_value = get_router_decision(user_question)
            except Exception as routing_error:
                execution_mode_key, routing_system_rationale, effective_k_value = "gemini", "Bypass execution fallback route.", 12
                logger.error(f"Router module runtime error trace: {routing_error}")

            # FIX: on the cloud there is no Ollama daemon, so any local route
            # from the router is transparently promoted to the Gemini tier.
            if IS_CLOUD and execution_mode_key in ("qwen", "ollama"):
                routing_system_rationale = (
                    f"{routing_system_rationale} "
                    "[Promoted to Gemini: local tiers unavailable on Streamlit Cloud]"
                )
                execution_mode_key, effective_k_value = "gemini", 12

            # The router can now pick an OpenAI (GPT) tier. If no OpenAI key is
            # configured, transparently fall back to Gemini (also a k=12 cloud
            # tier) so Auto-Pilot never dead-ends on a missing credential.
            if execution_mode_key in ("gpt4o", "gpt5") and not os.getenv("OPENAI_API_KEY"):
                routing_system_rationale = (
                    f"{routing_system_rationale} "
                    "[Promoted to Gemini: OPENAI_API_KEY not configured]"
                )
                execution_mode_key, effective_k_value = "gemini", 12

        # Engage execution layer
        try:
            generated_answer, source_evidence_docs = run_ui_query(mode=execution_mode_key, question=user_question, k=effective_k_value)
        except Exception as execution_error:
            generated_answer = f"System connection error occurred during model generation. Detail logs: {str(execution_error)}"
            source_evidence_docs = []

    # Commit full state trace package
    observability_state_package = {
        "mode": execution_mode_key,
        "k": effective_k_value,
        "router_reason": routing_system_rationale,
        "docs": source_evidence_docs
    }
    st.session_state.messages.append({"role": "assistant", "content": generated_answer, "meta": observability_state_package})
    render_system_turn(generated_answer)
    render_data_provenance(observability_state_package)
