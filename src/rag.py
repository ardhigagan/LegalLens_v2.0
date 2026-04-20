"""
rag.py — LegalLens AI v2: Chat with your Contract
- Embeds contract chunks into a FAISS vector store
- Retrieves top-k relevant chunks for each user question
- Generates cited answers using the Anthropic Claude API
- Falls back to a local HuggingFace model if no API key provided
"""

import os
import numpy as np

# Vector store
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not installed. Run: pip install faiss-cpu")

# Embeddings
from sentence_transformers import SentenceTransformer

# Anthropic Claude API
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# HuggingFace QA fallback
from transformers import pipeline as hf_pipeline
import torch

_embed_model = None
_qa_pipeline = None


def _get_embed_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _embed_model
    if _embed_model is None:
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _get_qa_pipeline():
    """Lazy-load a local HuggingFace QA pipeline as fallback."""
    global _qa_pipeline
    if _qa_pipeline is None:
        device = 0 if torch.cuda.is_available() else -1
        print("Loading local QA model (deepset/roberta-base-squad2)...")
        _qa_pipeline = hf_pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=device,
        )
    return _qa_pipeline


class ContractRAG:
    """
    Manages a FAISS vector index for a single contract session.
    Usage:
        rag = ContractRAG()
        rag.build_index(chunks)
        answer, citations = rag.ask("What are the termination conditions?")
    """

    def __init__(self):
        self.chunks = []
        self.index = None
        self.embed_model = _get_embed_model()
        self.dimension = 384  # all-MiniLM-L6-v2 output dim

    def build_index(self, chunks: list[str]) -> None:
        """Embeds all chunks and builds a FAISS index."""
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS is not installed. Run: pip install faiss-cpu")

        self.chunks = chunks
        print(f"Embedding {len(chunks)} chunks...")

        embeddings = self.embed_model.encode(chunks, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        print("FAISS index built.")

    def retrieve(self, question: str, top_k: int = 4) -> list[dict]:
        """
        Retrieves the top-k most relevant chunks for a question.
        Returns list of dicts with 'text' and 'score'.
        """
        if self.index is None or not self.chunks:
            return []

        q_embedding = self.embed_model.encode([question])
        q_embedding = np.array(q_embedding, dtype="float32")
        faiss.normalize_L2(q_embedding)

        scores, indices = self.index.search(q_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks) and score > 0.1:
                results.append({
                    "text": self.chunks[idx],
                    "score": float(score),
                    "chunk_id": int(idx),
                })
        return results

    def ask(self, question: str, api_key: str = None, top_k: int = 4) -> tuple[str, list[dict]]:
        """
        Answers a question about the contract.

        Args:
            question: The user's question.
            api_key: Groq API key (gsk_...) or Anthropic key (sk-ant-...).
                     If None, uses local QA model.
            top_k: Number of context chunks to retrieve.

        Returns:
            answer (str): The generated answer.
            citations (list[dict]): The retrieved chunks used as context.
        """
        citations = self.retrieve(question, top_k=top_k)

        if not citations:
            return "I couldn't find relevant clauses in this contract to answer your question.", []

        context = "\n\n---\n\n".join(
            [f"[Clause {i+1}]: {c['text']}" for i, c in enumerate(citations)]
        )

        # --- Option A: Groq API (free, fast, recommended) ---
        if api_key and api_key.startswith("gsk_"):
            answer = _answer_with_groq(question, context, api_key)

        # --- Option B: Anthropic Claude API ---
        elif api_key and api_key.startswith("sk-ant") and ANTHROPIC_AVAILABLE:
            answer = _answer_with_claude(question, context, api_key)

        # --- Option C: Local HuggingFace QA (fallback, no key needed) ---
        else:
            answer = _answer_with_local_model(question, context)

        return answer, citations


def _answer_with_groq(question: str, context: str, api_key: str) -> str:
    """Uses Groq's free API (llama-3.1-8b) to generate a cited answer."""
    try:
        import urllib.request
        import json as _json

        prompt = f"""You are a legal analyst assistant. Answer the user's question based ONLY on the contract clauses provided below.
Always cite which clause number you are referencing (e.g. "According to Clause 2...").
If the answer is not found in the clauses, say so clearly. Be concise and precise.

CONTRACT CLAUSES:
{context}

USER QUESTION: {question}

ANSWER:"""

        payload = _json.dumps({
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.2,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"Groq API error: {e}. Falling back — try the local model instead."


def _answer_with_claude(question: str, context: str, api_key: str) -> str:
    """Uses Anthropic Claude to generate a cited answer."""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"""You are a legal analyst assistant. Answer the user's question based ONLY on the contract clauses provided below. 
Always cite which clause number you are referencing (e.g., "According to Clause 2..."). 
If the answer is not found in the clauses, say so clearly.

CONTRACT CLAUSES:
{context}

USER QUESTION: {question}

ANSWER:"""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return f"Claude API error: {e}. Falling back to local model."


def _answer_with_local_model(question: str, context: str) -> str:
    """Uses a local extractive QA model as fallback."""
    try:
        qa = _get_qa_pipeline()
        # Truncate context to model's max input
        result = qa(question=question, context=context[:2000])
        confidence = int(result["score"] * 100)
        return (
            f"{result['answer']}\n\n"
            f"*(Extracted answer — confidence: {confidence}%. "
            f"For better answers, add an Anthropic API key in Settings.)*"
        )
    except Exception as e:
        return f"Could not generate an answer: {e}"
