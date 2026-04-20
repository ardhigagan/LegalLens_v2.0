"""
processing.py — Upgraded LegalLens AI v2
- Sentence-aware chunking using NLTK (no mid-clause splits)
- Falls back to RecursiveCharacterTextSplitter if NLTK unavailable
- chunk_size tuned for legal documents (512 tokens ≈ 1800 chars)
"""

import re

try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

from langchain_text_splitters import RecursiveCharacterTextSplitter


def _sentence_aware_chunks(text: str, chunk_size: int = 1800, overlap: int = 200) -> list[str]:
    """
    Splits text into chunks that always end on a sentence boundary.
    This prevents cutting mid-clause, which hurts classification accuracy.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_len + sentence_len > chunk_size and current:
            chunks.append(" ".join(current))
            # Overlap: keep last N chars worth of sentences
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current = overlap_sentences
            current_len = overlap_len

        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if len(c.strip()) > 50]


def _langchain_chunks(text: str, chunk_size: int = 1800, overlap: int = 200) -> list[str]:
    """Fallback: LangChain RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_text(text)


def chunk_text(text: str, chunk_size: int = 1800, chunk_overlap: int = 200) -> list[str]:
    """
    Main chunking function.
    Uses sentence-aware splitting when NLTK is available,
    falls back to LangChain splitter otherwise.
    """
    if not text:
        return []

    if NLTK_AVAILABLE:
        print("Using sentence-aware chunking (NLTK)...")
        chunks = _sentence_aware_chunks(text, chunk_size, chunk_overlap)
    else:
        print("Using LangChain character splitter (NLTK not available)...")
        chunks = _langchain_chunks(text, chunk_size, chunk_overlap)

    print(f"Split document into {len(chunks)} chunks.")
    return chunks
