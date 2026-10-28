"""Agent routing, corrective RAG control loop, grounded answer composition and guardrails."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .ollama import OllamaClient

from .retrieval import HybridRetriever, RetrievalResult, citations_from, expand_query, tokens
from .schemas import AgentRoute, QueryRequest, QueryResponse, RetrievalTrace


POLICY_TERMS = {"gpa", "probation", "degree", "graduation", "credit", "credits", "prerequisite", "prerequisites", "fee", "fees", "tuition", "deadline", "policy", "handbook", "regulation", "withdraw"}
RESEARCH_TERMS = {"paper", "papers", "literature", "research", "study", "studies", "citation", "arxiv", "methodology", "review"}
INTEGRITY_PATTERNS = (
    r"\b(do|solve|write|complete|answer)\b.*\b(my|this|the)\b.*\b(assignment|homework|exam|test|quiz)\b",
    r"\bsubmit.*\bfor me\b",
    r"\bcheat\b",
)
INJECTION_PATTERNS = (r"ignore (all |any |the )?(previous|above) instructions", r"reveal (your )?(system|hidden) prompt")


@dataclass(frozen=True)
class RoutedQuery:
    route: AgentRoute
    reason: str


def route_query(query: str) -> RoutedQuery:
    lowered = query.lower()
    if any(re.search(pattern, lowered) for pattern in INTEGRITY_PATTERNS):
        return RoutedQuery(AgentRoute.integrity, "The request appears to seek a directly submit-ready assessment answer.")
    words = set(tokens(lowered))
    if words & POLICY_TERMS or re.search(r"\b[A-Za-z]{2,5}\s?-?\d{2,4}\b", query):
        return RoutedQuery(AgentRoute.policy, "Institutional rules, requirements, or course identifiers were detected.")
    if words & RESEARCH_TERMS:
        return RoutedQuery(AgentRoute.research, "Research discovery or scholarly synthesis language was detected.")
    return RoutedQuery(AgentRoute.pedagogy, "Defaulted to the teaching and learning agent.")


def transform_query(query: str) -> str:
    """Deterministic query rewriter used when retrieval is not grounded enough."""
    stripped = re.sub(r"\b(please|can you|could you|i need|tell me|help me)\b", " ", query, flags=re.I)
    stripped = re.sub(r"\s+", " ", stripped).strip(" ?.")
    return expand_query(stripped)


def _most_relevant_sentences(query: str, result: RetrievalResult, limit: int = 4) -> list[tuple[str, int]]:
    query_terms = set(tokens(query))
    candidates: list[tuple[float, str, int]] = []
    for index, hit in enumerate(result.hits, start=1):
        sentences = re.split(r"(?<=[.!?])\s+", hit.chunk.parent_text)
        for sentence in sentences:
            clean = sentence.strip()
            if len(clean) < 25:
                continue
            sentence_terms = set(tokens(clean))
            overlap = len(query_terms & sentence_terms) / max(len(query_terms), 1)
            score = overlap + hit.rerank_score * 0.65
            candidates.append((score, clean, index))
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    selected: list[tuple[str, int]] = []
    seen: set[str] = set()
    for _, sentence, citation_index in candidates:
        key = sentence.lower()
        if key not in seen:
            selected.append((sentence, citation_index))
            seen.add(key)
        if len(selected) == limit:
            break
    return selected


def _follow_ups(route: AgentRoute) -> list[str]:
    if route == AgentRoute.policy:
        return ["Which programme and catalogue year apply to you?", "Would you like a prerequisite or degree-audit check?"]
    if route == AgentRoute.research:
        return ["What research question or discipline should I narrow to?", "Should I prioritize recent work, seminal papers, or a specific method?"]
    return ["What part would you like unpacked further?", "Would a short practice question or study plan help?"]


class AcademicAgent:
    """Bounded self-correcting RAG workflow with explicit, inspectable trace data."""

    def __init__(self, retriever: HybridRetriever, max_loops: int = 3, ollama: Optional[OllamaClient] = None) -> None:
        self.retriever = retriever
        self.max_loops = max_loops
        self.ollama = ollama

    def answer(self, request: QueryRequest) -> QueryResponse:
        route = route_query(request.query)
        trace_notes = [route.reason]
        if any(re.search(pattern, request.query, re.I) for pattern in INJECTION_PATTERNS):
            trace = RetrievalTrace(
                original_query=request.query,
                effective_query=request.query,
                loop_count=0,
                retrieval_quality="blocked",
                hybrid_alpha=0.0,
                used_fallback=False,
                notes=["Prompt-injection-like instruction was ignored; only the academic question can be processed."],
            )
            return QueryResponse(
                answer="I can help with an academic question, but I can’t follow instructions that ask me to override safeguards or disclose hidden system information.",
                route=AgentRoute.integrity,
                trace=trace,
            )
        if route.route == AgentRoute.integrity:
            trace = RetrievalTrace(
                original_query=request.query,
                effective_query=request.query,
                loop_count=0,
                retrieval_quality="guardrailed",
                hybrid_alpha=0.0,
                used_fallback=False,
                notes=trace_notes + ["Academic-integrity guardrail activated."],
            )
            return QueryResponse(
                answer=("I can’t produce a submit-ready answer for an assessed task. I can still help you learn: share your attempt, "
                        "the rubric, or the concept you’re stuck on, and I’ll give targeted hints, an explanation, or a comparable practice problem."),
                route=route.route,
                follow_up_questions=["What have you tried so far?", "Which concept or step is blocking you?"],
                trace=trace,
            )

        effective_query = request.query
        result = self.retriever.search(effective_query, request.role, request.top_k)
        loops = 1
        # A retrieved result needs both a meaningful top hit and sufficient cross-hit quality.
        while (not result.hits or result.quality < 0.20) and loops < self.max_loops:
            rewritten = self.ollama.rewrite_query(effective_query) if self.ollama else None
            effective_query = rewritten or transform_query(effective_query)
            if rewritten:
                model_name = self.ollama.model if self.ollama else "local"
                trace_notes.append(f"Low retrieval confidence triggered a {model_name} query rewrite and another retrieval pass.")
            else:
                trace_notes.append("Low retrieval confidence triggered a deterministic query transformation and another retrieval pass.")
            result = self.retriever.search(effective_query, request.role, request.top_k)
            loops += 1
        fallback = not result.hits or result.quality < 0.20
        if fallback:
            trace_notes.append("No sufficiently grounded internal context was found; external research fallback is intentionally not enabled in this local demo.")
            trace = RetrievalTrace(
                original_query=request.query,
                effective_query=effective_query,
                loop_count=loops,
                retrieval_quality="insufficient",
                hybrid_alpha=result.alpha,
                used_fallback=True,
                notes=trace_notes,
            )
            return QueryResponse(
                answer=("I couldn’t verify that answer against the academic knowledge base available to your role. "
                        "Try adding a course code, programme, policy name, or upload the relevant syllabus/handbook."),
                route=route.route,
                follow_up_questions=_follow_ups(route.route),
                trace=trace,
            )

        sentences = _most_relevant_sentences(request.query, result)
        claims = "\n".join(f"- {sentence} [{citation_index}]" for sentence, citation_index in sentences)
        opening = {
            AgentRoute.policy: "Based on the approved institutional material available to you:",
            AgentRoute.pedagogy: "Here’s a grounded explanation from the course material:",
            AgentRoute.research: "Here’s what the available scholarly material supports:",
        }[route.route]
        answer = f"{opening}\n{claims}" if claims else "I found relevant material, but not a precise enough passage to make a reliable claim."
        generated_answer = self.ollama.grounded_answer(request.query, route.route, result.hits) if self.ollama else None
        if generated_answer:
            answer = generated_answer
            model_name = self.ollama.model if self.ollama else "local"
            trace_notes.append(f"{model_name} produced a citation-validated answer from the retrieved sources.")
        elif self.ollama and self.ollama.enabled:
            trace_notes.append(f"{self.ollama.model} was unavailable or returned an uncited answer; used the citation-only fallback.")
        trace_notes.append("Reflection check passed: every presented claim is tied to a retrieved citation.")
        trace = RetrievalTrace(
            original_query=request.query,
            effective_query=effective_query,
            loop_count=loops,
            retrieval_quality="grounded",
            hybrid_alpha=result.alpha,
            used_fallback=False,
            notes=trace_notes,
        )
        return QueryResponse(
            answer=answer,
            route=route.route,
            citations=citations_from(result.hits),
            follow_up_questions=_follow_ups(route.route),
            trace=trace,
        )
