from pathlib import Path
import sys
from uuid import uuid4

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.app import app


client = TestClient(app)


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:8]}"


def test_health_and_dashboard():
    health = client.get("/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["status"] == "ok"
    assert "mode" in payload["rag_health"]
    assert payload["rag_health"]["total_chunks"] > 0

    root = client.get("/")
    assert root.status_code == 200
    assert "Post-Op Triage" in root.text


def test_patient_triage_flows():
    low_id = _uid("pytest-low")
    mod_id = _uid("pytest-mod")
    critical_id = _uid("pytest-critical")

    low = client.post(
        "/api/patient/submit",
        json={
            "patient_info": {"patient_id": low_id, "age": 55},
            "symptoms": "mild soreness only",
            "surgery_protocol": "knee arthroscopy",
            "medications": "ibuprofen",
        },
    )
    assert low.status_code == 200
    low_payload = low.json()
    assert low_payload["status"] == "plan_delivered"
    assert low_payload["risk_level"] == "Low"
    assert low_payload["next_action"] == "patient_follow_up"
    assert low_payload["patient_message"]

    moderate = client.post(
        "/api/patient/submit",
        json={
            "patient_info": {"patient_id": mod_id, "age": 61},
            "symptoms": "fever and redness around wound",
            "surgery_protocol": "knee arthroscopy",
            "medications": "ibuprofen",
        },
    )
    assert moderate.status_code == 200
    mod_payload = moderate.json()
    assert mod_payload["status"] == "doctor_review_required"
    assert mod_payload["risk_level"] == "Moderate"
    assert mod_payload["next_action"] == "doctor_review"

    critical = client.post(
        "/api/patient/submit",
        json={
            "patient_info": {"patient_id": critical_id, "age": 48},
            "symptoms": "heavy bleeding and difficulty breathing",
            "surgery_protocol": "abdominal surgery",
            "medications": "acetaminophen",
        },
    )
    assert critical.status_code == 200
    critical_payload = critical.json()
    assert critical_payload["status"] == "emergency_escalation"
    assert critical_payload["risk_level"] == "Critical"
    assert critical_payload["next_action"] == "emergency_care"


def test_rag_ingest_search_and_answer(tmp_path: Path):
    source = _uid("pytest-protocol")
    ingest = client.post(
        "/api/rag/ingest",
        json={
            "text": "Patients with knee replacement should avoid stairs for 6 weeks.",
            "source": source,
        },
    )
    assert ingest.status_code == 200
    ingest_payload = ingest.json()
    assert ingest_payload["chunks_added"] >= 1

    search = client.get("/api/rag/search", params={"q": "knee stairs recovery"})
    assert search.status_code == 200
    search_payload = search.json()
    assert any(
        item.get("source") == source or "stairs" in item.get("content", "").lower()
        for item in search_payload["results"]
    )

    answer = client.post(
        "/api/rag/answer",
        json={"query": "What should a knee replacement patient avoid?", "audience": "patient"},
    )
    assert answer.status_code == 200
    answer_payload = answer.json()
    assert answer_payload["answer"]
    assert answer_payload["sources"]
    assert answer_payload["used_chunks"]
