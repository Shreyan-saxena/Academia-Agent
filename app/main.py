"""FastAPI entrypoint for AcademiaAgent."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .agents import AcademicAgent
from .config import get_settings
from .corpus import AcademicCorpus
from .ingestion import can_manage_knowledge_base, ingest_upload, load_persisted_documents, persist_uploaded_document
from .ollama import OllamaClient
from .retrieval import HybridRetriever, citations_from
from .schemas import DegreeAuditRequest, DegreeAuditResponse, HealthResponse, QueryRequest, QueryResponse, UserRole

settings = get_settings()
corpus = AcademicCorpus()
retriever = HybridRetriever(corpus)
ollama = OllamaClient(
    base_url=settings.ollama_base_url,
    model=settings.ollama_model,
    timeout_seconds=settings.ollama_timeout_seconds,
    enabled=settings.ollama_enabled,
)
agent = AcademicAgent(retriever, settings.max_retrieval_loops, ollama=ollama)


@asynccontextmanager
async def lifespan(_: FastAPI):
    corpus.load_seed_directory(settings.seed_dir)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    for document in load_persisted_documents(settings.upload_dir, client=ollama):
        corpus.add(document)
    yield


app = FastAPI(
    title="AcademiaAgent",
    version="0.1.0",
    description="Multi-agent, self-correcting RAG for higher education.",
    lifespan=lifespan,
)
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", documents=corpus.document_count, chunks=corpus.chunk_count)


@app.get("/api/documents", tags=["knowledge-base"])
def list_documents():
    data = [doc.model_dump() for doc in corpus.documents()]
    return JSONResponse(
        content=data,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.post("/api/documents", status_code=201, tags=["knowledge-base"])
async def upload_document(
    file: Annotated[UploadFile, File(description="Markdown, text, PDF, or DOCX academic source")],
    uploader_role: Annotated[UserRole, Form(description="Role of the user adding this shared knowledge-base source")],
    roles: Annotated[str, Form(description="Comma-separated roles permitted to retrieve this material")] = "student,faculty,staff,administrator",
):
    if not can_manage_knowledge_base(uploader_role):
        raise HTTPException(403, "Only administrator, staff and faculty are able to add the file to the knowledge base.")
    allowed_roles: list[UserRole] = []
    for role in roles.split(","):
        try:
            allowed_roles.append(UserRole(role.strip()))
        except ValueError as error:
            raise HTTPException(422, f"Unknown role: {role}") from error
    document = await ingest_upload(file, settings.upload_dir, allowed_roles, client=ollama)
    persist_uploaded_document(document, settings.upload_dir)
    corpus.add(document)
    return {"id": document.id, "title": document.title, "message": "Document ingested and available for retrieval."}


@app.post("/api/query", response_model=QueryResponse, tags=["academic-agent"])
def query(request: QueryRequest) -> QueryResponse:
    return agent.answer(request)


@app.post("/api/degree-audit", response_model=DegreeAuditResponse, tags=["policy-agent"])
def degree_audit(request: DegreeAuditRequest) -> DegreeAuditResponse:
    """A transparent starter degree audit, deliberately advisory rather than authoritative."""
    completed = {course.upper().replace(" ", "") for course in request.completed_courses}
    outstanding: list[str] = []
    eligible: list[str] = []
    if request.earned_credits < 120:
        outstanding.append(f"{120 - request.earned_credits:g} more credits are needed to reach the 120-credit minimum.")
    if request.cumulative_gpa < 2.0:
        outstanding.append("Cumulative GPA is below the 2.0 minimum described in the handbook.")
    if "CS201" not in completed:
        outstanding.append("CS201 is not recorded as completed; it is required before CS401.")
    if "MATH202" not in completed:
        outstanding.append("MATH202 is not recorded as completed; it is required before CS401.")
    if {"CS201", "MATH202"}.issubset(completed):
        eligible.append("CS401 (subject to seat availability and any programme-specific rules)")
    else:
        eligible.append("Complete CS201 and MATH202 before planning CS401.")
    policy_hits = retriever.search("degree requirements 120 credits GPA CS401 prerequisites", UserRole.student, 3)
    citations = citations_from(policy_hits.hits)
    status = "on track" if not outstanding else "needs review"
    return DegreeAuditResponse(
        status=status,
        eligible_courses=eligible,
        outstanding_requirements=outstanding or ["No gap was detected by the starter rules."],
        advisory=("This is an advisory screening based on the seeded handbook rules, not an official degree clearance. "
                  "Confirm substitutions, transfer credits, and catalogue-year requirements with your registrar."),
        citations=citations,
    )


static_dir = settings.root_dir / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(static_dir / "index.html")
