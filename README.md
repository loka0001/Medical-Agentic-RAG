# Agentic RAG Post-Op Triage Support System

AI-assisted post-operative triage and recovery follow-up system.

## What this system is

- A FastAPI application for post-operative triage support
- A workflow that helps route patients into low-risk guidance, doctor review, or emergency escalation
- A recovery follow-up tool that stores case history and supports doctor feedback
- A document-grounded RAG tool for uploaded protocols and recovery guidance

## What this system is not

- Not a diagnosis system
- Not a treatment engine
- Not an autonomous doctor
- Not a production medical device
- Not GraphRAG
- Not a complex hospital EMR

## Core product flow

1. Start the server
2. Open the dashboard
3. Upload or ingest protocols if needed
4. Submit a patient update
5. Read the triage result
   - Low -> plan delivered
   - Moderate/High -> doctor review required
   - Critical -> emergency escalation
6. Doctor reviews pending cases if needed
7. Patient sends follow-up updates after a plan is delivered
8. Use RAG search or grounded answers to inspect uploaded protocols

## How RAG works

`RagService` supports four retrieval modes:

- `faiss_openai`
- `faiss_local`
- `tfidf_vector`
- `keyword_emergency`

The service keeps:

- built-in post-operative guidance
- manually ingested text
- uploaded TXT, MD, and PDF text

RAG is used in two places:

1. `MediatorAgent` decision support
2. Direct document-grounded Q&A through `/api/rag/answer`

Uploaded entries are persisted in:

- `data/rag_entries.json`

## Upload behavior

Supported uploads:

- `.txt`
- `.md`
- `.pdf`

If a PDF has no extractable text:

- the API still returns `200`
- `chunks_added` will be `0`
- the response includes:
  `No extractable text found. Scanned PDFs require OCR, which is not enabled.`

This project does not include OCR.

## Fallback behavior

- If cloud LLM access fails, patient-facing and structuring agents fall back safely
- If cloud embeddings fail, RAG falls back to local vector retrieval or TF-IDF
- `keyword_emergency` is the last-resort fallback only

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Dashboard |
| GET | `/health` | App + RAG + storage health |
| GET | `/api/system/state` | Pending cases, RAG health, storage counts |
| POST | `/api/patient/submit` | Submit a new patient update |
| POST | `/api/doctor/feedback` | Submit doctor review for a pending case |
| POST | `/api/patient/feedback` | Submit patient follow-up |
| POST | `/api/rag/ingest` | Ingest manual text into RAG |
| POST | `/api/rag/upload` | Upload TXT/MD/PDF into RAG |
| GET | `/api/rag/search` | Search knowledge chunks |
| POST | `/api/rag/answer` | Ask a grounded question from retrieved chunks |
| GET | `/api/rag/documents` | Show indexed and stored documents |
| GET | `/api/cases/{patient_id}` | Show patient history |

## How to run

```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn api.app:app --reload
```

Open:

- [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

## Example requests

### Patient submit

```json
POST /api/patient/submit
{
  "patient_info": {
    "patient_id": "test-low",
    "age": 56
  },
  "symptoms": "mild soreness only",
  "surgery_protocol": "knee arthroscopy",
  "medications": "ibuprofen"
}
```

### Doctor feedback

```json
POST /api/doctor/feedback
{
  "patient_id": "test-mod",
  "instructions": "Reviewed. Continue monitoring temperature and wound changes.",
  "risk_override": "Moderate"
}
```

### RAG ingest

```json
POST /api/rag/ingest
{
  "text": "Patients with knee replacement should avoid stairs for 6 weeks.",
  "source": "knee-protocol"
}
```

### RAG answer

```json
POST /api/rag/answer
{
  "query": "What should a knee replacement patient avoid?",
  "audience": "patient"
}
```

## Limitations

- This is triage support only and not a substitute for professional medical advice
- Direct cloud LLM calls depend on valid credentials and working network access
- Blank or scanned PDFs require OCR, which is not enabled
- RAG answers are grounded support summaries and not diagnosis or prescriptions

## Disclaimer

This system provides triage support and recovery follow-up assistance only. It is not a substitute for professional medical advice, diagnosis, or treatment.
