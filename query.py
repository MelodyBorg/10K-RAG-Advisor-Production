#!/usr/bin/env python3
"""
query.py
========
Enterprise-grade Dispatcher and RAG Execution Engine.
Supports multi-tier inference over SEC 10-K filings.

Integration Contracts fulfilled:
    - run_ui_query(mode, question, k) -> (answer_text, retrieved_docs)
    - Terminal-only CLI loop mode via main()

FIXES vs previous version:
    1. GoogleGenaiEmbeddings (nonexistent class) -> GoogleGenerativeAIEmbeddings
    2. Embedding model "text-embedding-004" -> "models/text-embedding-004"
       (must match build_index.py exactly, or retrieval silently degrades)
    3. ChatOllama model "qwen" -> "qwen3" (run `ollama pull qwen3`, or set
       OLLAMA_QWEN_MODEL to match your local tag from `ollama list`)
    4. Prefer langchain_ollama imports (community versions are deprecated)
    5. Exceptions now chain their root cause so app.py logs the real error
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

# LangChain and Core Imports
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Setup Logging to match enterprise terminal visibility
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
logger = logging.getLogger("query")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Override these if `ollama list` shows different tags on your machine,
# e.g. OLLAMA_QWEN_MODEL=qwen3:8b or qwen2.5:7b
QWEN_MODEL = os.getenv("OLLAMA_QWEN_MODEL", "qwen3:8b")
MISTRAL_MODEL = os.getenv("OLLAMA_MISTRAL_MODEL", "mistral")

# ---------------------------------------------------------------------------
# Embedding and Model Initialization Dispatchers
# ---------------------------------------------------------------------------

def get_embeddings(mode: str):
    """
    Resolves the correct vector embedding space to prevent dimension mismatches.
    - Gemini and Qwen share the Gemini/Google embedding space.
    - Mistral (Ollama) uses the local Nomic text space.
    """
    if mode in ("gemini", "qwen"):
        # FIX 1: correct class name (GoogleGenaiEmbeddings does not exist)
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        # FIX 2: model id needs the "models/" prefix and must match build_index.py
        return GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    elif mode == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings  # current package
        except ImportError:
            from langchain_community.embeddings import OllamaEmbeddings  # legacy fallback
        return OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
    else:
        raise ValueError(f"Unknown embedding registry mode: {mode}")


def get_llm(mode: str):
    """
    Binds the exact language model instance according to the selected tier.
    Optimizes local models with a 16k context window block constraint.
    """
    if mode == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    elif mode == "qwen":
        ChatOllama = _import_chat_ollama()
        # FIX 3: "qwen" is not a valid local tag; use qwen3:8b (or your pulled tag)
        return ChatOllama(
            model=QWEN_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            num_ctx=16384,  # Expanded safeguard window
        )

    elif mode == "ollama":
        ChatOllama = _import_chat_ollama()
        return ChatOllama(
            model=MISTRAL_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
            num_ctx=16384,  # Expanded safeguard window
        )
    else:
        raise ValueError(f"Unknown LLM tier identifier: {mode}")


def _import_chat_ollama():
    """FIX 4: prefer the maintained langchain_ollama package."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        from langchain_community.chat_models import ChatOllama
    return ChatOllama


def load_index(index_dir: Path, embeddings) -> FAISS:
    """Safely loads the target vector directory with proper path assertion."""
    if not index_dir.exists():
        raise FileNotFoundError(f"Vector directory not found at: {index_dir.resolve()}")
    logger.info("Loading FAISS index from %s ...", index_dir.resolve())
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)

# ---------------------------------------------------------------------------
# Premium UI / Programmatic Entrypoint Contract
# ---------------------------------------------------------------------------

def run_ui_query(mode: str, question: str, k: int, base_index_dir: str = ".") -> tuple[str, list[Document]]:
    """
    Executes a single stateless RAG pipeline sweep with Entity-Partitioned Retrieval
    to prevent cross-company context crowding.
    """
    # 1. Coordinate index architecture pathways dynamically
    if mode in ("gemini", "qwen"):
        target_path = Path(base_index_dir) / "faiss_index_gemini"
    else:
        target_path = Path(base_index_dir) / "faiss_index_ollama"

    # 2. Extract engine parts
    try:
        embeddings = get_embeddings(mode)
    except ImportError as exc:
        raise RuntimeError(
            f"Embedding backend import failed for mode '{mode}': {exc}. "
            "Check installed packages (langchain-google-genai / langchain-ollama)."
        ) from exc

    store = load_index(target_path, embeddings)
    llm = get_llm(mode)

    # 3. Entity-Partitioned Execution Sweep
    logger.info("Querying Tier [%s] with k=%d against index [%s]", mode, k, target_path.name)
    
    # Detect target companies mentioned in the prompt
    detected_entities = []
    lower_question = question.lower()
    if "alphabet" in lower_question or "google" in lower_question:
        detected_entities.append("Alphabet")
    if "amazon" in lower_question:
        detected_entities.append("Amazon")
    if "microsoft" in lower_question:
        detected_entities.append("Microsoft")

    docs = []
    # If multiple companies are mentioned, divide k and query them separately
    if len(detected_entities) > 1:
        k_per_entity = max(1, k // len(detected_entities))
        logger.info(f"Detected entities {detected_entities}. Partitioning retrieval to k={k_per_entity} per entity.")
        
        for entity in detected_entities:
            # Filter strictly by the 'company' key stamped by your build_index.py metadata
            entity_docs = store.similarity_search(
                question, 
                k=k_per_entity, 
                filter={"company": entity}
            )
            docs.extend(entity_docs)
    else:
        # Fallback to normal global search if it's a single-company or general query
        docs = store.similarity_search(question, k=k)

    # 4. Standard Context Generation Sequence
    context = "\n\n".join(
        f"--- Chunk from {d.metadata.get('company', 'Unknown')} "
        f"({d.metadata.get('year', '?')}, {d.metadata.get('section', 'Unknown section')}) ---\n"
        f"{d.page_content}"
        for d in docs
    )

    prompt = ChatPromptTemplate.from_template(
        "You are an expert financial analyst analyzing SEC 10-K filings.\n"
        "Use ONLY the following retrieved chunks to answer the user's question. "
        "If the answer cannot be confidently deduced from the text, explicitly state "
        "that you cannot answer using the provided 10-K data.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    chain = prompt | llm | StrOutputParser()
    try:
        answer_text = chain.invoke({"context": context, "question": question})
    except Exception as exc:
        raise RuntimeError(
            f"Generation failed on tier '{mode}': {exc}. "
            f"If local: verify the Ollama daemon is running and the model tag "
            f"is pulled."
        ) from exc

    return answer_text, docs
  

# ---------------------------------------------------------------------------
# Retro CLI Terminal Loop Compatibility Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="10-K RAG Engine Core CLI Client")
    parser.add_argument("--mode", type=str, default="gemini", choices=["gemini", "qwen", "ollama"])
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--index-dir", type=str, default=".")
    args = parser.parse_args()

    print("\n=== 10-K RAG CLI Engine Active ===")
    print(f"Mode: {args.mode} | Target K: {args.k}\nType 'exit' or 'quit' to leave.\n")

    while True:
        try:
            question = input("Question > ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                print("Goodbye.")
                break

            ans, docs = run_ui_query(
                mode=args.mode, question=question, k=args.k,
                base_index_dir=args.index_dir,
            )

            print("\n" + "=" * 80 + "\nFINAL ANSWER\n" + "=" * 80)
            print(ans)
            print("=" * 80 + "\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            logger.error("CLI Cycle Error: %s", e)


if __name__ == "__main__":
    main()
