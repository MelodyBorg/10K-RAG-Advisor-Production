---
title: 10K RAG Advisor
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# 10-K RAG Advisor TECH NOTES

A multi-model RAG chatbot that compares **Alphabet, Amazon, and Microsoft** using their FY2025 SEC 10-K filings. Ask it anything from "How much cash did Amazon have at the end of 2025?" to "How does Microsoft's cloud strategy differ from Amazon's?" — it retrieves the relevant filing passages, answers strictly from them, and shows you its sources.

---

## 1. Approach

### Architecture

```
User question (Streamlit chat UI — app.py)
        │
        ▼
Semantic Router (router.py, gpt-4o-mini, strict JSON output)
        │  classifies the question into one of four tiers
        ▼
Execution Engine (query.py)
        │  entity-partitioned FAISS retrieval → grounded prompt → tier LLM
        ▼
Answer + provenance (model used, routing reason, retrieved chunks & pages)
```

Three design principles:

1. **Right model for the right question.** Cross-company synthesis goes to a big cloud model; a single number lookup doesn't need one.
2. **Grounding over fluency.** The generation prompt forbids answering beyond the retrieved chunks, and every answer ships with its source chunks so claims can be audited.
3. **Fail soft, tell the user.** Every external dependency (router API, model APIs, local Ollama daemon) has a fallback path, and failures produce a clear user-facing explanation instead of a stack trace.

### Model choice

| Tier | Model | Used for | k |
|------|-------|----------|---|
| Auto-Pilot | gpt-4o-mini (router) | Classifies each query into one of the tiers below | — |
| Cloud / Advanced | **Gemini 2.5 Flash** | Cross-company comparisons, multi-hop synthesis | 12 |
| Cloud / Advanced | **GPT-5.4** | Single-company causal/strategic "why" analysis | 12 |
| Cloud (manual) | **GPT-4o** | Side-by-side quality comparison against GPT-5.4 | 12 |
| Local / Fast | **Qwen3 8B** (Ollama) | Single-company number/date lookups | 4 |
| Local / Narrative | **Mistral 7B** (Ollama) | Single-company disclosure/footnote reads | 4 |

The router itself runs on **gpt-4o-mini** deliberately: it is fast, costs fractions of a cent, and enforces a strict JSON schema — and critically, it does **not** consume the Gemini quota that the main Gemini tier depends on (see *Insights* below for the failure that taught us this).

### System prompts

- **Generation prompt** (all tiers): the model acts as an *expert financial analyst* and must answer **only** from the retrieved 10-K chunks; if the answer cannot be confidently deduced from them, it must explicitly say so. This is the primary hallucination control.
- **Router prompt**: a deterministic classifier prompt with tier definitions, a disambiguation table (multiple companies → Gemini; one company + "why/analyze" → GPT-5.4; one company + number → Qwen3; one company + "read/summarize" → Mistral), and few-shot examples. Temperature 0, JSON-schema-enforced output.

---

## 2. RAG configuration

| Component | Setting |
|-----------|---------|
| Chunking | `RecursiveCharacterTextSplitter`, **chunk_size = 1200**, **overlap = 300** (25%) |
| Separators | `["\n\n", "\n", ". ", " ", ""]` — paragraph first, protecting single-line financial table rows |
| Embeddings (cloud index) | Gemini `gemini-embedding-2` → `faiss_index_gemini` |
| Embeddings (local index) | Ollama `nomic-embed-text` → `faiss_index_ollama` |
| Vector store | **FAISS**, persisted in-repo, loaded per query |
| Chunk metadata | `company`, `year`, 10-K `section` (e.g. *Item 1A. Risk Factors*, detected by regex), `page`, `chunk_id` |

A key insight we exploit: **the embedding model and the generation LLM are independent.** The GPT-4o, GPT-5.4, and Qwen3 tiers all query the *Gemini-embedded* index — adding a new LLM tier requires zero re-embedding. Only retrieval must use the same embedding space the index was built with.

---

## 3. The four key engineering elements

### 3.1 Larger chunks so tables survive retrieval

We initially chunked at 800 characters and financial tables kept getting **cut off mid-row** — a retrieval hit would contain the line-item labels but not the numbers (or 2025's column but not 2024's), making number questions unanswerable. We increased the chunk size to **1200 with a 300 overlap** and put paragraph/newline separators before sentence splits, so multi-row balance-sheet excerpts stay intact inside a single chunk.

### 3.2 Fair-share (entity-partitioned) retrieval

Naive top-k retrieval over all three filings has a crowding problem: for "Compare Amazon and Microsoft's cloud revenue," plain similarity search might return 10 Amazon chunks and 2 Microsoft chunks — starving the model of one side of the comparison. We detect which companies a question mentions and **split k evenly per company** using FAISS metadata filters: a k=12 two-company question retrieves **6 + 6**, a three-company question 4 + 4 + 4. Every entity gets a fair share of the context window.

### 3.3 Dockerized deployment with the local models baked in

Streamlit Community Cloud cannot run Ollama, which would have limited a public demo to cloud APIs only. Instead we **dockerized the entire app**: the image installs the Ollama runtime, and `start.sh` boots the daemon, pulls `qwen3:8b` and `mistral`, then launches Streamlit. A GitHub Action (`.github/workflows/sync.yml`) syncs the repo to a **Hugging Face Space** running that Docker image — giving a public link where every tier except the OpenAI ones works (we don't expose our OpenAI key on the public deployment).

### 3.4 The router assistant (Auto-Pilot)

Instead of making the user guess which model suits their question, the default **Auto-Pilot** mode classifies each query in ~1s and dispatches it to the right tier with the right retrieval depth. The routing decision — tier, reason, and k — is shown in the *Data Provenance* panel of every answer, so routing is transparent rather than magic. If the router itself fails (rate limit, missing key, network), it falls back to Gemini and the UI states exactly why.

---

## 4. Insights & lessons learned (including what failed)

- **Routing on your scarcest resource is a death spiral.** Our first router ran on `gemini-2.5-flash` — the same model as our main answer tier, sharing the free tier's 5 requests/min, 20/day quota. Router calls competed with answers; once the quota ran out, the router's *exception fallback silently returned "gemini"* for everything, so Auto-Pilot appeared to "disproportionately choose Gemini" while actually being broken. The fix: route on gpt-4o-mini (a different provider's cheap model) and give every fallback a human-readable reason surfaced in the UI. Lesson: **silent fallbacks mask failures — make degradation visible.**
- **The LangChain ecosystem moves fast.** `ChatOllama` was removed from `langchain_community` 0.4.x (the package is being sunset), which broke the local tiers with an import error mid-project. Fix: the standalone `langchain-ollama` package. Pin your dependency assumptions.
- **Streamlit ate our dollar signs.** Markdown treats `$...$` as LaTeX math — financial answers full of dollar amounts got mangled into italic gibberish. Fix: render `$` as the HTML entity `&#36;`.
- **Reasoning models have quirks.** Qwen3 emits `<think>...</think>` reasoning blocks that must be handled; GPT-5.x models reject some sampling parameters that older models accept. Test each model's raw output, don't assume interchangeability.
- **Graceful degradation is a feature.** The app probes the Ollama daemon and hides the local tiers when it's absent (with an explanatory caption) rather than letting queries fail; missing API keys degrade Auto-Pilot to available tiers with a visible notice. The "reliable product" mindset shaped as much code as the RAG itself.

## 5. Strengths & weaknesses

**Strengths**
- Accurate, source-cited number extraction (e.g., Amazon FY2025 cash: **$86,810M**, cited to the Consolidated Balance Sheets page) — with local Qwen3 and cloud tiers agreeing, which we use as a cross-check.
- Balanced multi-model routing keeps costs low and answers fast without sacrificing quality on hard questions.
- Robust: every failure mode we found produces a clear user-facing message, not a stack trace.
- Extensible: new LLM tiers reuse the existing index (no re-embedding).

**Weaknesses**
- Free-tier Gemini quota still caps sustained cross-company usage.
- Entity detection is string-based ("Amazon", "Google"…) — a question like "compare the three companies" without naming them won't trigger fair-share partitioning.
- Only FY2025 filings are indexed; multi-year trend questions rely on the prior-year comparatives inside the FY2025 10-Ks.
- Derived metrics (e.g., margins) require the model to compute from retrieved numbers — a residual hallucination risk we mitigate with the grounding prompt but cannot eliminate.
- Local models are noticeably weaker on multi-hop reasoning; the router keeps such questions away from them, but manual tier selection can expose it.

## 6. Team

| Name | Role |
|------|------|
| Xing Wang | Added Open-AI model, Refined backend (router, exception handling, etc.), Helped with debugging, Tested Model |
| Melody Borg | Deployed qwen mistral Gemini, Built router, Fix chunk amount retrieval, Deployed dockerized on hugging face |
| Erhao Shao | Concluded and summarized the slides |
| Fernando Erler | Drafted Questions, Created and organized presentation |
| Zhezhen Wang | Explained retrieval bias in RAG systems and present our Fair Share Retrieval solution |
| Tianwen Guan | Drafted 10 questions for testing |

## 7. Running locally

```bash
pip install -r requirements.txt

# Keys: create .streamlit/secrets.toml (gitignored) with
#   GOOGLE_API_KEY = "..."
#   OPENAI_API_KEY = "..."

# Local tiers (optional): install Ollama, then
ollama serve &
ollama pull qwen3:8b && ollama pull mistral && ollama pull nomic-embed-text

# (Re)build the indexes if needed
python build_index.py --embedding gemini --data-dir ./PDFs
python build_index.py --embedding ollama --data-dir ./PDFs

streamlit run app.py
```

Or with Docker (as deployed on the Hugging Face Space):

```bash
docker build -t rag-advisor . && docker run -p 8501:8501 rag-advisor
```
