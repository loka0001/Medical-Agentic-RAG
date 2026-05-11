import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import faiss
except Exception:
    faiss = None

try:
    from langchain_community.vectorstores import FAISS as LangchainFAISS
except Exception:
    LangchainFAISS = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

load_dotenv()


class RagService:
    KNOWLEDGE_BASE = [
        "Heavy bleeding after surgery is an emergency and needs immediate medical care.",
        "Uncontrolled bleeding should not wait for routine follow-up.",
        "Difficulty breathing or being unable to breathe after surgery requires emergency escalation.",
        "Chest pain after surgery requires emergency medical evaluation.",
        "Fainting, loss of consciousness, or seizure after surgery should be treated as an emergency.",
        "Fever after surgery may need doctor review, especially when combined with wound changes.",
        "Redness, warmth, swelling, or discharge from the wound may need doctor review.",
        "Pus or a foul smell from the wound should be reviewed urgently.",
        "A wound that opens after surgery should not wait for routine follow-up.",
        "Calf pain, calf swelling, or calf redness after surgery may need urgent review.",
        "Persistent vomiting after surgery may need urgent medical review.",
        "Patients should not change, stop, skip, or double medications without clinician guidance.",
        "Missed medication doses should be reported rather than corrected with extra doses unless advised.",
        "Pain medication should be taken only as prescribed by the treating clinician.",
        "Routine post-operative care includes following wound care instructions and monitoring symptoms.",
        "Mild stable soreness without red-flag symptoms may be appropriate for routine monitoring.",
        "Patients should attend scheduled follow-up appointments even when symptoms appear mild.",
        "Rapidly worsening symptoms should prompt urgent or emergency care depending on severity.",
        "This system supports triage and recovery follow-up communication only and does not diagnose or treat.",
    ]

    def __init__(self):
        self.entries_path = Path("data/rag_entries.json")
        self.entries_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = []
        self.splitter = self._build_splitter()
        self.last_error = None
        self.openai_embeddings = self._build_openai_embeddings()
        self.local_model = self._build_local_model()
        self.model = self._build_model()
        self.openai_vectorstore = None
        self.local_index = None
        self.local_matrix = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.mode = "keyword_emergency"
        self._load_entries()
        self._ensure_built_in_entries()
        self._rebuild_indexes()

    def ingest_text(self, text, source="manual"):
        chunks = self._split_text(text)
        if not chunks:
            return {"source": source, "chunks_added": 0, "mode": self.mode, "message": "No text content was provided."}
        self._append_chunks(chunks, source=source, doc_type="text", path=None)
        self._save_entries()
        self._rebuild_indexes()
        return {"source": source, "chunks_added": len(chunks), "mode": self.mode}

    def ingest_file(self, file_path, source=None):
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix not in {".pdf", ".txt", ".md"}:
            raise ValueError("Unsupported file type.")
        doc_type = suffix.lstrip(".")
        if suffix == ".pdf":
            if PdfReader is None:
                raise ValueError("PDF support is unavailable because pypdf is not installed.")
            reader = PdfReader(str(path))
            text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
            if not text:
                return {
                    "source": source or path.name,
                    "chunks_added": 0,
                    "mode": self.mode,
                    "message": "No extractable text found. Scanned PDFs require OCR, which is not enabled.",
                    "path": str(path),
                }
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = self._split_text(text)
        if not chunks:
            return {
                "source": source or path.name,
                "chunks_added": 0,
                "mode": self.mode,
                "message": "No extractable text found in the file.",
                "path": str(path),
            }
        self._append_chunks(chunks, source=source or path.name, doc_type=doc_type, path=str(path))
        self._save_entries()
        self._rebuild_indexes()
        return {"source": source or path.name, "chunks_added": len(chunks), "mode": self.mode, "path": str(path)}

    def search(self, query, k=3):
        clean = (query or "").strip()
        if not clean:
            return []
        if self.mode == "faiss_openai" and self.openai_vectorstore is not None:
            try:
                results = self.openai_vectorstore.similarity_search_with_score(clean, k=k)
                return [
                    self._format_result(
                        content=doc.page_content,
                        metadata=doc.metadata or {},
                        score=float(score),
                    )
                    for doc, score in results
                ]
            except Exception as exc:
                self.last_error = f"openai_search_failed: {exc}"
        if self.mode == "faiss_local" and self.local_model is not None and self.local_matrix is not None:
            try:
                query_vector = np.array(self.local_model.encode([clean], normalize_embeddings=True), dtype="float32")
                if self.local_index is not None:
                    scores, indices = self.local_index.search(query_vector, min(k, len(self.entries)))
                    results = []
                    for score, index in zip(scores[0], indices[0]):
                        if index < 0:
                            continue
                        results.append(self._format_result_from_entry(self.entries[index], float(score)))
                    if results:
                        return results
                scores = (self.local_matrix @ query_vector[0]).tolist()
                return self._top_results(scores, k)
            except Exception as exc:
                self.last_error = f"local_vector_search_failed: {exc}"
        if self.mode == "tfidf_vector" and self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
            try:
                query_vector = self.tfidf_vectorizer.transform([clean])
                scores = cosine_similarity(query_vector, self.tfidf_matrix).ravel().tolist()
                return self._top_results(scores, k)
            except Exception as exc:
                self.last_error = f"tfidf_search_failed: {exc}"
        return self._keyword_search(clean, k)

    def answer(self, query: str, audience: str = "doctor") -> dict:
        results = self.search(query, k=5)
        if not results:
            return {
                "query": query,
                "audience": audience,
                "answer": "No relevant knowledge was found.",
                "sources": [],
                "used_chunks": [],
                "mode": self.mode,
            }
        sources = [
            {
                "source": item.get("source", "unknown"),
                "score": item.get("score"),
                "content_preview": self._preview(item.get("content", "")),
            }
            for item in results
        ]
        used_chunks = [
            {
                "source": item.get("source", "unknown"),
                "score": item.get("score"),
                "content": item.get("content", ""),
            }
            for item in results
        ]
        answer = self._fallback_answer(results, audience)
        if self.model is not None:
            try:
                prompt = (
                    "You are generating document-grounded post-operative support only. "
                    "Use only the provided retrieved text. Do not diagnose, prescribe, or invent facts. "
                    "Always mention that this is document-grounded support and not a diagnosis. "
                    f"Audience: {audience}. "
                    "If the audience is patient, use simple safe language. If the audience is doctor, use concise clinical-style language.\n"
                    f"Question: {query}\n"
                    f"Retrieved text: {json.dumps([item['content'] for item in results], ensure_ascii=False)}"
                )
                answer = self._text(self.model.invoke(prompt)) or answer
            except Exception as exc:
                self.last_error = f"rag_answer_failed: {exc}"
        return {
            "query": query,
            "audience": audience,
            "answer": answer,
            "sources": sources,
            "used_chunks": used_chunks,
            "mode": self.mode,
        }

    def list_documents(self):
        grouped = {}
        for entry in self.entries:
            source = entry.get("source", "unknown")
            grouped.setdefault(
                source,
                {
                    "source": source,
                    "chunks": 0,
                    "doc_type": entry.get("doc_type"),
                    "created_at": entry.get("created_at"),
                    "path": entry.get("path"),
                },
            )
            grouped[source]["chunks"] += 1
            if entry.get("created_at") and (
                not grouped[source]["created_at"] or entry["created_at"] > grouped[source]["created_at"]
            ):
                grouped[source]["created_at"] = entry["created_at"]
            if entry.get("path"):
                grouped[source]["path"] = entry["path"]
        return [grouped[source] for source in sorted(grouped, key=str.lower)]

    def health(self):
        return {
            "status": "ok",
            "mode": self.mode,
            "total_chunks": len(self.entries),
            "documents": len(self.list_documents()),
            "last_error": self.last_error,
            "vector_available": self.mode != "keyword_emergency",
        }

    def _build_splitter(self):
        if RecursiveCharacterTextSplitter is None:
            return None
        return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    def _build_openai_embeddings(self):
        if os.getenv("OPENAI_API_KEY"):
            try:
                return OpenAIEmbeddings(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                )
            except Exception as exc:
                self.last_error = f"openai_embedding_init_failed: {exc}"
                return None
        if os.getenv("GITHUB_TOKEN"):
            try:
                return OpenAIEmbeddings(
                    api_key=os.getenv("GITHUB_TOKEN"),
                    model=os.getenv("GITHUB_EMBEDDING_MODEL", "text-embedding-3-small"),
                    base_url="https://models.inference.ai.azure.com",
                )
            except Exception as exc:
                self.last_error = f"github_embedding_init_failed: {exc}"
                return None
        return None

    def _build_local_model(self):
        if SentenceTransformer is None:
            return None
        try:
            return SentenceTransformer(os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"), local_files_only=True)
        except Exception as exc:
            self.last_error = f"local_embedding_init_failed: {exc}"
            return None

    def _build_model(self):
        if os.getenv("OPENAI_API_KEY"):
            try:
                return ChatOpenAI(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    api_key=os.getenv("OPENAI_API_KEY"),
                    temperature=0,
                )
            except Exception:
                return None
        if os.getenv("GITHUB_TOKEN"):
            try:
                return ChatOpenAI(
                    model=os.getenv("GITHUB_MODEL", "gpt-4o"),
                    api_key=os.getenv("GITHUB_TOKEN"),
                    base_url="https://models.inference.ai.azure.com",
                    temperature=0,
                )
            except Exception:
                return None
        return None

    def _load_entries(self):
        if not self.entries_path.exists():
            return
        try:
            loaded = json.loads(self.entries_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                self.entries = [entry for entry in loaded if isinstance(entry, dict) and entry.get("content")]
        except Exception as exc:
            self.last_error = f"rag_entries_load_failed: {exc}"
            self.entries = []

    def _save_entries(self):
        try:
            self.entries_path.write_text(json.dumps(self.entries, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            self.last_error = f"rag_entries_save_failed: {exc}"

    def _ensure_built_in_entries(self):
        existing = {
            (entry.get("source"), entry.get("content"))
            for entry in self.entries
            if entry.get("source") == "built-in-guidance"
        }
        changed = False
        for index, snippet in enumerate(self.KNOWLEDGE_BASE):
            key = ("built-in-guidance", snippet)
            if key in existing:
                continue
            self.entries.append(
                {
                    "content": snippet,
                    "source": "built-in-guidance",
                    "doc_type": "built_in",
                    "created_at": None,
                    "chunk_index": index,
                }
            )
            changed = True
        if changed:
            self._save_entries()

    def _append_chunks(self, chunks, source, doc_type, path=None):
        timestamp = self._now()
        for index, chunk in enumerate(chunks):
            self.entries.append(
                {
                    "content": chunk,
                    "source": source,
                    "doc_type": doc_type,
                    "path": path,
                    "created_at": timestamp,
                    "chunk_index": index,
                }
            )

    def _split_text(self, text):
        clean = (text or "").strip()
        if not clean:
            return []
        if self.splitter is not None:
            return [chunk.strip() for chunk in self.splitter.split_text(clean) if chunk.strip()]
        return [clean[index:index + 500].strip() for index in range(0, len(clean), 450) if clean[index:index + 500].strip()]

    def _rebuild_indexes(self):
        self.openai_vectorstore = None
        self.local_index = None
        self.local_matrix = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.mode = "keyword_emergency"
        texts = [entry["content"] for entry in self.entries if entry.get("content")]
        metadata = [self._entry_metadata(entry) for entry in self.entries if entry.get("content")]
        if not texts:
            return
        if self.openai_embeddings is not None and LangchainFAISS is not None:
            try:
                self.openai_vectorstore = LangchainFAISS.from_texts(texts, self.openai_embeddings, metadatas=metadata)
                self.mode = "faiss_openai"
                return
            except Exception as exc:
                self.last_error = f"openai_vector_build_failed: {exc}"
        if self.local_model is not None:
            try:
                matrix = self.local_model.encode(texts, normalize_embeddings=True)
                self.local_matrix = np.array(matrix, dtype="float32")
                if faiss is not None and len(texts) > 0:
                    self.local_index = faiss.IndexFlatIP(self.local_matrix.shape[1])
                    self.local_index.add(self.local_matrix)
                self.mode = "faiss_local"
                return
            except Exception as exc:
                self.last_error = f"local_vector_build_failed: {exc}"
                self.local_index = None
                self.local_matrix = None
        try:
            self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            self.mode = "tfidf_vector"
            return
        except Exception as exc:
            self.last_error = f"tfidf_build_failed: {exc}"
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
        self.mode = "keyword_emergency"

    def _top_results(self, scores, k):
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        results = []
        for index, score in ranked[:k]:
            if score <= 0:
                continue
            results.append(self._format_result_from_entry(self.entries[index], float(score)))
        return results or self._keyword_search("", k)

    def _keyword_search(self, query, k):
        terms = [term for term in re.findall(r"[\w\u0600-\u06FF]+", query.lower()) if len(term) > 1]
        scored = []
        for entry in self.entries:
            haystack = f"{entry.get('source', '')} {entry.get('content', '')}".lower()
            score = sum(haystack.count(term) for term in terms)
            if score:
                scored.append(self._format_result_from_entry(entry, float(score)))
        scored.sort(key=lambda item: item.get("score", 0), reverse=True)
        if scored:
            return scored[:k]
        return [self._format_result_from_entry(entry, None) for entry in self.entries[:k]]

    def _format_result_from_entry(self, entry, score):
        return self._format_result(entry.get("content", ""), self._entry_metadata(entry), score)

    def _format_result(self, content, metadata, score):
        result = {
            "content": content,
            "source": metadata.get("source", "unknown"),
            "doc_type": metadata.get("doc_type"),
            "created_at": metadata.get("created_at"),
            "chunk_index": metadata.get("chunk_index"),
            "path": metadata.get("path"),
        }
        if score is not None:
            result["score"] = score
        return result

    def _entry_metadata(self, entry):
        return {
            "source": entry.get("source", "unknown"),
            "doc_type": entry.get("doc_type"),
            "path": entry.get("path"),
            "created_at": entry.get("created_at"),
            "chunk_index": entry.get("chunk_index"),
        }

    def _fallback_answer(self, results, audience):
        top_score = results[0].get("score") if results and results[0].get("score") is not None else None
        selected = results[:3]
        if top_score and top_score > 0:
            threshold = top_score * 0.25
            selected = [item for item in results if (item.get("score") or 0) >= threshold][:3] or results[:3]
        snippets = list(
            dict.fromkeys(
                self._clean_answer_chunk(item.get("content", ""))
                for item in selected
                if item.get("content")
            )
        )
        if audience == "patient":
            return "Document-grounded patient support, not a diagnosis: " + " ".join(snippets)
        return "Document-grounded support for doctor review, not a diagnosis: " + " ".join(snippets)

    def _preview(self, text, limit=160):
        clean = " ".join((text or "").split())
        if len(clean) <= limit:
            return clean
        return clean[: limit - 3].rstrip() + "..."

    def _clean_answer_chunk(self, text):
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        cleaned = " ".join(line.lstrip("#").strip() for line in lines)
        return " ".join(cleaned.split())

    def _now(self):
        return datetime.now(timezone.utc).isoformat()

    def _text(self, response):
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content).strip()
