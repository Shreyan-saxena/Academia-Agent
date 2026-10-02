from app.agents import AcademicAgent, route_query
from app.corpus import AcademicCorpus, AcademicDocument, load_markdown
from app.ingestion import can_manage_knowledge_base, load_persisted_documents, persist_uploaded_document
from app.retrieval import HybridRetriever
from app.schemas import AgentRoute, QueryRequest, UserRole


def make_agent() -> AcademicAgent:
    corpus = AcademicCorpus()
    corpus.load_seed_directory(__import__("pathlib").Path("data/seed"))
    return AcademicAgent(HybridRetriever(corpus))


def test_policy_question_is_grounded_and_cited():
    response = make_agent().answer(QueryRequest(
        query="What are the prerequisites for CS401?",
        role=UserRole.student,
    ))
    assert response.route == AgentRoute.policy
    assert response.trace.retrieval_quality == "grounded"
    assert response.citations
    assert "CS201" in response.answer


def test_assessment_request_is_guardrailed():
    response = make_agent().answer(QueryRequest(
        query="Solve my homework assignment for me",
        role=UserRole.student,
    ))
    assert response.route == AgentRoute.integrity
    assert response.trace.retrieval_quality == "guardrailed"
    assert not response.citations


def test_unknown_question_does_not_invent_answer():
    response = make_agent().answer(QueryRequest(
        query="What is the shuttle timetable to the north campus tomorrow?",
        role=UserRole.student,
    ))
    assert response.trace.retrieval_quality == "insufficient"
    assert "couldn’t verify" in response.answer


def test_research_route():
    assert route_query("How should I evaluate papers in a literature review?").route == AgentRoute.research


class CitationReadyOllama:
    model = "gemma4:12b"
    enabled = True

    def rewrite_query(self, query):
        return None

    def grounded_answer(self, query, route, hits):
        assert hits
        return "CS401 requires CS201 and MATH202 with a minimum grade of C. [1]"


def test_local_llm_answer_is_used_only_after_citation_validation():
    corpus = AcademicCorpus()
    corpus.load_seed_directory(__import__("pathlib").Path("data/seed"))
    agent = AcademicAgent(HybridRetriever(corpus), ollama=CitationReadyOllama())
    response = agent.answer(QueryRequest(query="What are the prerequisites for CS401?", role=UserRole.student))
    assert response.answer.endswith("[1]")
    assert any("gemma4:12b produced a citation-validated answer" in note for note in response.trace.notes)


def test_uploaded_document_is_restored_from_the_local_manifest(tmp_path):
    document = AcademicDocument(
        id="upload_persisted",
        title="Faculty Handbook Addendum",
        source="upload:faculty-handbook.txt",
        text="The Faculty Handbook Addendum contains a durable rule for the academic knowledge base.",
        allowed_roles=(UserRole.faculty, UserRole.staff),
    )
    persist_uploaded_document(document, tmp_path)
    restored = load_persisted_documents(tmp_path)
    assert restored == [document]
    assert (tmp_path / "knowledge-base.json").exists()


def test_only_institutional_staff_can_manage_the_knowledge_base():
    assert not can_manage_knowledge_base(UserRole.student)
    assert can_manage_knowledge_base(UserRole.faculty)
    assert can_manage_knowledge_base(UserRole.staff)
    assert can_manage_knowledge_base(UserRole.administrator)


def test_unmanifested_disk_files_are_automatically_indexed_and_persisted(tmp_path):
    # Simulate a file placed in upload_dir without an entry in knowledge-base.json
    raw_file = tmp_path / "research_syllabus-abcd1234.md"
    raw_file.write_text("# Research Syllabus\n\nThis syllabus outlines advanced deep learning methodologies and evaluation techniques for students.", encoding="utf-8")
    
    # Load documents from directory: should automatically detect, extract, and write manifest
    restored = load_persisted_documents(tmp_path)
    assert len(restored) == 1
    assert restored[0].title == "Research Syllabus"
    assert "advanced deep learning" in restored[0].text
    assert (tmp_path / "knowledge-base.json").exists()


def test_upload_endpoint_role_restrictions(tmp_path, monkeypatch):
    import dataclasses
    from fastapi.testclient import TestClient
    from app.main import app, settings

    new_settings = dataclasses.replace(settings, upload_dir=tmp_path)
    monkeypatch.setattr("app.main.settings", new_settings)
    client = TestClient(app)
    
    # Student attempt should fail with 403
    response_student = client.post(
        "/api/documents",
        data={"uploader_role": "student", "roles": "student,faculty"},
        files={"file": ("notes.txt", b"These are lecture notes with sufficient content to extract.", "text/plain")},
    )
    assert response_student.status_code == 403
    assert "Only administrator, staff and faculty are able to add the file to the knowledge base." in response_student.json()["detail"]

    # Faculty attempt should succeed with 201
    response_faculty = client.post(
        "/api/documents",
        data={"uploader_role": "faculty", "roles": "student,faculty"},
        files={"file": ("faculty_notes.txt", b"Faculty notes with sufficient content to extract and index.", "text/plain")},
    )
    assert response_faculty.status_code == 201
    assert "Document ingested" in response_faculty.json()["message"]
