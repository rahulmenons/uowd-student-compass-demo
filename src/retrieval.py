"""Transparent UOWD source retrieval using either a lexical or a dense embedding model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "uowd_sources.jsonl"


def load_documents(path: Path | str = DATA_PATH) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        documents = [json.loads(line) for line in f if line.strip()]
    if not documents:
        raise ValueError(f"No documents available in {path}")
    if len({d["id"] for d in documents}) != len(documents):
        raise ValueError("Document ids must be unique")
    return documents


class TfidfRetriever:
    """Strong, cheap baseline with keyword + bigram matching."""

    def __init__(self, documents: list[dict[str, Any]]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            lowercase=True, strip_accents="unicode", stop_words="english",
            ngram_range=(1, 2), sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform([d["title"] + ". " + d["text"] for d in documents])

    def search(self, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        question = question.strip()
        if not question:
            return []
        vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(vector, self.matrix)[0]
        order = np.argsort(-similarities, kind="stable")[:top_k]
        return [{**self.documents[int(i)], "score": float(similarities[i])} for i in order]


class MiniLMRetriever:
    """Neural semantic retrieval; downloads a pretrained model on first use."""

    def __init__(self, documents: list[dict[str, Any]], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("Install optional neural dependencies using: pip install -r requirements.txt") from error
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        # Use identical source data for fair retrieval comparison.
        self.matrix = self.model.encode(
            [d["title"] + ". " + d["text"] for d in documents],
            normalize_embeddings=True, show_progress_bar=False,
        )

    def search(self, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        question = question.strip()
        if not question:
            return []
        query_vector = self.model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
        similarities = np.asarray(self.matrix) @ np.asarray(query_vector)
        order = np.argsort(-similarities, kind="stable")[:top_k]
        return [{**self.documents[int(i)], "score": float(similarities[i])} for i in order]


def recommend_answer(results: list[dict[str, Any]], threshold: float) -> dict[str, Any] | None:
    """Conservative extractive answer, never a fabricated generated answer.

    Threshold is a demo heuristic; calibrate it on a validation set before deploying.
    The score is similarity, NOT model accuracy or probability of correctness.
    """
    if not results or results[0]["score"] < threshold:
        return None
    return results[0]
