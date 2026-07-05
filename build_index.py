#!/usr/bin/env python3
"""
build_index.py
==============
Data ingestion + FAISS vector index builder for a 10-K comparison RAG system
(Alphabet, Amazon, Microsoft).

Pipeline:
    1. Load 10-K PDFs from a data directory (LangChain PyPDFLoader).
    2. Extract metadata (company, fiscal year) from filenames and tag chunks.
    3. Detect standard SEC 10-K section headers (e.g., "Item 1A. Risk Factors")
       and attach the section name to each chunk's metadata.
    4. Chunk with RecursiveCharacterTextSplitter tuned for financial text
       (chunk_size=800, 15% overlap, separators preserving table rows).
    5. Embed with either Gemini or Ollama (nomic-embed-text), selected via CLI.
    6. Persist a FAISS index locally (./faiss_index_<model>).

Usage:
    python build_index.py --embedding gemini --data-dir ./data
    python build_index.py --embedding ollama --data-dir ./data

Environment:
    GOOGLE_API_KEY   required for --embedding gemini
    OLLAMA_BASE_URL  optional, defaults to http://localhost:11434

Dependencies:
    pip install langchain langchain-community langchain-google-genai \
                pypdf faiss-cpu langchain-ollama
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_index")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300  

# Splitting priority: paragraph breaks first, then line breaks (protects
# financial table rows, which are single lines), then sentence ends, then
# whitespace, then hard character split as a last resort.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Known company aliases -> canonical names (extend as needed)
COMPANY_ALIASES = {
    "alphabet": "Alphabet",
    "goog": "Alphabet",
    "googl": "Alphabet",
    "google": "Alphabet",
    "amazon": "Amazon",
    "amzn": "Amazon",
    "microsoft": "Microsoft",
    "msft": "Microsoft",
}

# Regex for standard 10-K section headers, e.g.:
#   "ITEM 1A. RISK FACTORS", "Item 7. Management's Discussion..."
SECTION_HEADER_RE = re.compile(
    r"^\s*ITEM\s+(\d{1,2}[A-C]?)\s*[.:\-]?\s*(.{0,120}?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Canonical 10-K section titles used to clean up noisy header captures
CANONICAL_SECTIONS = {
    "1": "Item 1. Business",
    "1A": "Item 1A. Risk Factors",
    "1B": "Item 1B. Unresolved Staff Comments",
    "1C": "Item 1C. Cybersecurity",
    "2": "Item 2. Properties",
    "3": "Item 3. Legal Proceedings",
    "4": "Item 4. Mine Safety Disclosures",
    "5": "Item 5. Market for Registrant's Common Equity",
    "6": "Item 6. Selected Financial Data / Reserved",
    "7": "Item 7. Management's Discussion and Analysis (MD&A)",
    "7A": "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Item 8. Financial Statements and Supplementary Data",
    "9": "Item 9. Changes in and Disagreements with Accountants",
    "9A": "Item 9A. Controls and Procedures",
    "9B": "Item 9B. Other Information",
    "9C": "Item 9C. Disclosure Regarding Foreign Jurisdictions",
    "10": "Item 10. Directors, Executive Officers and Corporate Governance",
    "11": "Item 11. Executive Compensation",
    "12": "Item 12. Security Ownership",
    "13": "Item 13. Certain Relationships and Related Transactions",
    "14": "Item 14. Principal Accountant Fees and Services",
    "15": "Item 15. Exhibits and Financial Statement Schedules",
    "16": "Item 16. Form 10-K Summary",
}


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------
def parse_filename_metadata(pdf_path: Path) -> dict:
    """
    Extract {company, year} from a filename like:
        Microsoft_10K_2025.pdf, amzn-10k-2024.pdf, Alphabet 10-K FY2023.pdf
    Falls back to 'Unknown' when a field can't be determined.
    """
    stem = pdf_path.stem.lower()

    company = "Unknown"
    for alias, canonical in COMPANY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", stem) or alias in stem:
            company = canonical
            break

    year_match = re.search(r"(20\d{2})", stem)
    year = year_match.group(1) if year_match else "Unknown"

    if company == "Unknown":
        logger.warning("Could not infer company from filename: %s", pdf_path.name)
    if year == "Unknown":
        logger.warning("Could not infer year from filename: %s", pdf_path.name)

    return {"company": company, "year": year, "source_file": pdf_path.name}


def normalize_section(item_number: str, raw_title: str) -> str:
    """Map a detected 'Item X' header to a canonical section title."""
    key = item_number.upper()
    if key in CANONICAL_SECTIONS:
        return CANONICAL_SECTIONS[key]
    title = raw_title.strip().rstrip(".").title()
    return f"Item {key}. {title}" if title else f"Item {key}"


def annotate_sections(pages: list[Document]) -> None:
    """
    Walk pages in order, tracking the most recently seen 10-K section header,
    and stamp each page's metadata with 'section'. Header detection is
    heuristic: matches lines beginning with 'Item <n>' at line start.
    Pages before the first detected header (cover, TOC) get 'Front Matter'.

    Note: the table of contents also lists items; to reduce false positives we
    only accept a header if the matched line is short (a real header, not a
    paragraph mentioning an item) and appears after page 3.
    """
    current_section = "Front Matter"
    for i, page in enumerate(pages):
        text = page.page_content or ""
        # Find the LAST plausible header on this page so subsequent pages
        # inherit the newest section.
        found: Optional[str] = None
        for m in SECTION_HEADER_RE.finditer(text):
            line = m.group(0).strip()
            # Filter out ToC noise / inline references:
            if i <= 2:
                continue  # skip cover + ToC pages
            if len(line) > 130:
                continue
            found = normalize_section(m.group(1), m.group(2))
            # If the header appears early in the page, the whole page belongs
            # to the new section; tag the page itself with it.
            if m.start() < 300:
                current_section = found
        page.metadata["section"] = current_section
        if found and found != current_section:
            # Header appeared mid/late page; next pages belong to it.
            current_section = found


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def load_documents(data_dir: Path) -> list[Document]:
    """Load every PDF in data_dir, tagging pages with filename + section metadata."""
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found in %s", data_dir.resolve())
        sys.exit(1)

    all_pages: list[Document] = []
    for pdf_path in pdf_files:
        logger.info("Loading %s ...", pdf_path.name)
        file_meta = parse_filename_metadata(pdf_path)

        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        annotate_sections(pages)
        for page in pages:
            page.metadata.update(file_meta)

        logger.info(
            "  -> %d pages | company=%s | year=%s",
            len(pages), file_meta["company"], file_meta["year"],
        )
        all_pages.extend(pages)

    logger.info("Loaded %d total pages from %d PDF file(s).",
                len(all_pages), len(pdf_files))
    return all_pages


def chunk_documents(pages: list[Document]) -> list[Document]:
    """Split pages into retrieval-sized chunks preserving table/paragraph structure."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)

    # Give each chunk a stable id useful for eval/debugging.
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = (
            f"{chunk.metadata.get('company','NA')}-"
            f"{chunk.metadata.get('year','NA')}-{idx:05d}"
        )

    logger.info(
        "Created %d chunks (size=%d, overlap=%d, separators=%s).",
        len(chunks), CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS[:3],
    )
    return chunks


# ---------------------------------------------------------------------------
# Embeddings factory (experimentation switch)
# ---------------------------------------------------------------------------
def get_embeddings(name: str):
    """
    Return an embeddings object for 'gemini' or 'ollama'.
    Imports are deferred so users only need the packages they actually use.
    """
    name = name.lower()
    if name == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            logger.error("GOOGLE_API_KEY environment variable is not set.")
            sys.exit(1)
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError:
            logger.error("Missing package: pip install langchain-google-genai")
            sys.exit(1)
        logger.info("Using Gemini embeddings (models/gemini-embedding-2).")
        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        
    if name == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            try:  # older fallback
                from langchain_community.embeddings import OllamaEmbeddings
            except ImportError:
                logger.error("Missing package: pip install langchain-ollama")
                sys.exit(1)
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        logger.info("Using Ollama embeddings (nomic-embed-text @ %s).", base_url)
        return OllamaEmbeddings(model="nomic-embed-text", base_url=base_url)

    logger.error("Unknown embedding provider: %s (use 'gemini' or 'ollama')", name)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------
def build_faiss_index(chunks: list[Document], embeddings, out_dir: Path,
                      batch_size: int = 100) -> None:
    """Embed chunks in batches and persist the FAISS index to out_dir."""
    logger.info("Embedding %d chunks (batch_size=%d)...", len(chunks), batch_size)

    vector_store = None
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)
            #time.sleep(65)  # Wait 65 seconds after every batch to reset the 1-minute quota
        logger.info("  Embedded %d / %d chunks",
                    min(start + batch_size, len(chunks)), len(chunks))

    out_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(out_dir))
    logger.info("FAISS index saved successfully to %s "
                "(load later with FAISS.load_local).", out_dir.resolve())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FAISS index over 10-K PDFs for RAG."
    )
    parser.add_argument(
        "--embedding",
        choices=["gemini", "ollama"],
        required=True,
        help="Embedding backend to use.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Directory containing the 10-K PDF files (default: ./data).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir for the FAISS index "
             "(default: ./faiss_index_<embedding>).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Embedding batch size (default: 100).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or Path(f"./faiss_index_{args.embedding}")

    logger.info("=== 10-K RAG Index Builder ===")
    logger.info("Embedding backend : %s", args.embedding)
    logger.info("Data directory    : %s", args.data_dir.resolve())
    logger.info("Output directory  : %s", out_dir.resolve())

    if not args.data_dir.exists():
        logger.error("Data directory does not exist: %s", args.data_dir)
        sys.exit(1)

    pages = load_documents(args.data_dir)
    chunks = chunk_documents(pages)
    embeddings = get_embeddings(args.embedding)
    build_faiss_index(chunks, embeddings, out_dir, batch_size=args.batch_size)

    logger.info("Done. %d documents -> %d chunks -> index at %s",
                len(pages), len(chunks), out_dir)


if __name__ == "__main__":
    main()
