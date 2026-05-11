import sys
from enum import Enum
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator import SystemOrchestrator
from schemas import DoctorFeedback, PatientFeedback, PatientInput


class RagIngestRequest(BaseModel):
    text: str
    source: str | None = None


class RagAnswerRequest(BaseModel):
    query: str
    audience: str = "doctor"


app = FastAPI(title="Agentic RAG Post-Op Triage Support System", version="1.0.0")
orchestrator = SystemOrchestrator()


@app.get("/")
async def root():
    return FileResponse(ROOT / "system_flow_ui.html")


@app.get("/health")
async def health():
    try:
        rag_health = orchestrator.rag.health()
        return _dump(
            {
                "status": "ok",
                "rag_mode": rag_health["mode"],
                "rag_last_error": rag_health["last_error"],
                "rag_health": rag_health,
                "storage_summary": orchestrator.storage.state_summary(),
            }
        )
    except Exception as exc:
        return _error("health", exc)


@app.get("/api/system/state")
async def system_state():
    try:
        return _dump(orchestrator.state())
    except Exception as exc:
        return _error("system_state", exc)


@app.post("/api/patient/submit")
async def submit_patient(patient_input: PatientInput):
    try:
        return _dump(orchestrator.process_raw_input(patient_input))
    except Exception as exc:
        return _error("submit_patient", exc)


@app.post("/api/doctor/feedback")
async def doctor_feedback(feedback: DoctorFeedback):
    try:
        return _dump(orchestrator.process_doctor_feedback(feedback))
    except Exception as exc:
        return _error("doctor_feedback", exc)


@app.post("/api/patient/feedback")
async def patient_feedback(feedback: PatientFeedback):
    try:
        return _dump(orchestrator.process_patient_feedback(feedback))
    except Exception as exc:
        return _error("patient_feedback", exc)


@app.post("/api/rag/ingest")
async def rag_ingest(request: RagIngestRequest):
    try:
        return _dump(orchestrator.ingest_text(request.text, source=request.source or "manual"))
    except Exception as exc:
        return _error("rag_ingest", exc)


@app.post("/api/rag/answer")
async def rag_answer(request: RagAnswerRequest):
    try:
        return _dump(orchestrator.answer_from_knowledge(request.query, audience=request.audience or "doctor"))
    except Exception as exc:
        return _error("rag_answer", exc)


@app.post("/api/rag/upload")
async def rag_upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    try:
        uploads = ROOT / "data" / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        target = uploads / f"{uuid4().hex}_{Path(file.filename).name}"
        target.write_bytes(await file.read())
        return _dump(orchestrator.ingest_file(target, source=file.filename))
    except HTTPException:
        raise
    except Exception as exc:
        return _error("rag_upload", exc)


@app.get("/api/rag/search")
async def rag_search(q: str):
    try:
        return _dump(orchestrator.search_knowledge(q))
    except Exception as exc:
        return _error("rag_search", exc)


@app.get("/api/rag/documents")
async def rag_documents():
    try:
        return _dump(
            {
                "rag_documents": orchestrator.rag.list_documents(),
                "stored_documents": orchestrator.storage.list_documents(),
            }
        )
    except Exception as exc:
        return _error("rag_documents", exc)


@app.get("/api/cases/{patient_id}")
async def case_history(patient_id: str):
    try:
        return _dump(orchestrator.get_patient_history(patient_id))
    except Exception as exc:
        return _error("case_history", exc)


def _error(where, exc):
    return {"status": "error", "message": str(exc), "where": where}


def _dump(value):
    if isinstance(value, BaseModel):
        return _dump(value.model_dump())
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _dump(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_dump(item) for item in value]
    return value
