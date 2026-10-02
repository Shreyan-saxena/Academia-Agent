"""Hybrid retrieval and lightweight reranking with no external service requirement.

The interfaces intentionally mirror a Qdrant + BM25 deployment so they can be
swapped for production adapters without changing agents or HTTP contracts.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Iterable, Sequence

from .corpus import AcademicCorpus, Chunk
from .schemas import Citation, UserRole


TOKEN = re.compile(r"[A-Za-z]+(?:[0-9]+)?|\d+(?:\.\d+)?")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "do", "for", "from", "how", "i", "in", "is",
    "it", "me", "my", "of", "on", "or", "please", "the", "to", "what", "when", "with", "would", "you",
}
SYNONYMS = {
    "prerequisite": "requires required before",
    "prerequisites": "requires required before",
    "eligible": "eligibility requirements qualify",
    "graduation": "degree credits requirements complete",
    "probation": "academic gpa standing warning",
    "deadline": "date due calendar",
    "fees": "tuition payment financial",
    "help": "explain guidance",
}


def tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN.findall(text) if token.lower() not in STOP_WORDS]


def expand_query(query: str) -> str:
    additions = [SYNONYMS[word] for word in tokens(query) if word in SYNONYMS]
    return " ".join([query, *additions])


def is_exact_query(query: str) -> bool:
    return bool(re.search(r"\b[A-Za-z]{2,5}\s?-?\d{2,4}\b", query)) or any(
        phrase in query.lower() for phrase in ("policy", "handbook", "gpa", "deadline", "fee", "regulation")
    )


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    dense_score: float
    sparse_score: float
    hybrid_score: float
    rerank_score: float


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[RetrievedChunk]
    alpha: float

    @property
    def quality(self) -> float:
        if not self.hits:
            return 0.0
        return sum(hit.rerank_score for hit in self.hits[:3]) / min(len(self.hits), 3)


class HybridRetriever:
    def __init__(self, corpus: AcademicCorpus) -> None:
        self.corpus = corpus

    def search(self, query: str, role: UserRole, top_k: int = 5) -> RetrievalResult:
        candidates = self.corpus.chunks_for(role)
        if not candidates:
            return RetrievalResult(hits=[], alpha=0.5)
        expanded = expand_query(query)
        query_tokens = tokens(expanded)
        alpha = 0.35 if is_exact_query(query) else 0.72
        dense_scores = [self._cosine(query_tokens, tokens(chunk.text)) for chunk in candidates]
        sparse_scores = self._bm25(query_tokens, candidates)
        preliminary: list[RetrievedChunk] = []
        for chunk, dense, sparse in zip(candidates, dense_scores, sparse_scores):
            hybrid = alpha * dense + (1 - alpha) * sparse
            preliminary.append(RetrievedChunk(chunk, dense, sparse, hybrid, 0.0))
        preliminary.sort(key=lambda item: item.hybrid_score, reverse=True)
        # A compact cross-encoder stand-in: rewards coverage of content-bearing query terms
        reranked = [self._rerank(query_tokens, hit, alpha) for hit in preliminary[: min(25, len(preliminary))]]
        reranked.sort(key=lambda item: item.rerank_score, reverse=True)
        return RetrievalResult(hits=reranked[:top_k], alpha=alpha)

    @staticmethod
    def _cosine(query_tokens: Sequence[str], document_tokens: Sequence[str]) -> float:
        if not query_tokens or not document_tokens:
            return 0.0
        left, right = Counter(query_tokens), Counter(document_tokens)
        dot = sum(left[token] * right.get(token, 0) for token in left)
        norm_left = math.sqrt(sum(value * value for value in left.values()))
        norm_right = math.sqrt(sum(value * value for value in right.values()))
        return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0

    @staticmethod
    def _bm25(query_tokens: Sequence[str], chunks: Sequence[Chunk]) -> list[float]:
        documents = [tokens(chunk.text) for chunk in chunks]
        average_length = sum(len(document) for document in documents) / max(len(documents), 1)
        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(set(document))
        scores: list[float] = []
        for document in documents:
            frequencies = Counter(document)
            score = 0.0
            for term in query_tokens:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                idf = math.log(1 + (len(documents) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
                denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * len(document) / max(average_length, 1))
                score += idf * (frequency * 2.5) / denominator
            scores.append(score)
        maximum = max(scores, default=0.0)
        return [score / maximum if maximum else 0.0 for score in scores]

    @staticmethod
    def _rerank(query_tokens: Sequence[str], hit: RetrievedChunk, alpha: float) -> RetrievedChunk:
        document_terms = set(tokens(hit.chunk.text))
        coverage = sum(term in document_terms for term in set(query_tokens)) / max(len(set(query_tokens)), 1)
        phrase_bonus = 0.10 if any(term.isdigit() or re.search(r"[a-z]+\d+", term) for term in query_tokens) and coverage else 0.0
        score = min(1.0, 0.55 * hit.hybrid_score + 0.45 * coverage + phrase_bonus)
        return RetrievedChunk(hit.chunk, hit.dense_score, hit.sparse_score, hit.hybrid_score, score)


def citations_from(hits: Iterable[RetrievedChunk]) -> list[Citation]:
    citations: list[Citation] = []
    for hit in hits:
        excerpt = re.sub(r"\s+", " ", hit.chunk.text).strip()
        citations.append(Citation(
            document_id=hit.chunk.document_id,
            title=hit.chunk.title,
            section=hit.chunk.section,
            excerpt=excerpt[:320] + ("…" if len(excerpt) > 320 else ""),
            score=round(hit.rerank_score, 3),
        ))
    return citations
