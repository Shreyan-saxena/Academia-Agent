"""API contracts for AcademiaAgent."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class UserRole(str, Enum):
    student = "student"
    faculty = "faculty"
    staff = "staff"
    administrator = "administrator"


class AgentRoute(str, Enum):
    policy = "policy"
    pedagogy = "pedagogy"
    research = "research"
    integrity = "integrity"


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=4000)
    role: UserRole = UserRole.student
    course: Optional[str] = Field(None, max_length=30)
    top_k: int = Field(5, ge=1, le=10)


class Citation(BaseModel):
    document_id: str
    title: str
    section: str
    excerpt: str
    score: float


class RetrievalTrace(BaseModel):
    original_query: str
    effective_query: str
    loop_count: int
    retrieval_quality: str
    hybrid_alpha: float
    used_fallback: bool
    notes: List[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    route: AgentRoute
    citations: List[Citation] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    trace: RetrievalTrace


class DocumentSummary(BaseModel):
    id: str
    title: str
    source: str
    allowed_roles: List[UserRole]
    chunks: int


class DegreeAuditRequest(BaseModel):
    completed_courses: List[str] = Field(default_factory=list)
    earned_credits: float = Field(..., ge=0, le=300)
    cumulative_gpa: float = Field(..., ge=0, le=4.0)


class DegreeAuditResponse(BaseModel):
    status: str
    eligible_courses: List[str]
    outstanding_requirements: List[str]
    advisory: str
    citations: List[Citation]


class HealthResponse(BaseModel):
    status: str
    documents: int
    chunks: int

