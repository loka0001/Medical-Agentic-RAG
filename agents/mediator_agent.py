import json
import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from schemas import MediatorDecision, RecoveryPlan, RecoveryPlanStep, RiskLevel, StructuredPatientData

load_dotenv()


class MediatorAgent:
    CRITICAL_TERMS = [
        "heavy bleeding",
        "uncontrolled bleeding",
        "difficulty breathing",
        "cannot breathe",
        "chest pain",
        "loss of consciousness",
        "fainting",
        "seizure",
        "نزيف شديد",
        "صعوبة تنفس",
        "ألم صدر",
        "إغماء",
    ]
    HIGH_TERMS = [
        "pus",
        "foul smell",
        "wound opened",
        "severe pain",
        "persistent vomiting",
        "calf pain",
        "calf swelling",
        "صديد",
        "رائحة كريهة",
        "ألم شديد",
        "فتح الجرح",
    ]
    MODERATE_TERMS = [
        "fever",
        "redness",
        "swelling",
        "discharge",
        "increasing pain",
        "warm wound",
        "missed dose",
        "حمى",
        "سخونية",
        "احمرار",
        "تورم",
        "إفرازات",
    ]
    RANK = {RiskLevel.LOW: 1, RiskLevel.MODERATE: 2, RiskLevel.HIGH: 3, RiskLevel.CRITICAL: 4}

    def __init__(self, rag_service=None):
        self.rag = rag_service
        self.model = self._build_model()
        self.last_context = self._empty_context()

    def process_structured_data(self, data: StructuredPatientData) -> MediatorDecision:
        combined = self._combined_text(data)
        base_risk = self._screen_risk(combined)
        context_results = self._retrieve_context_results(combined, k=3)
        context = [item.get("content", "").strip() for item in context_results if item.get("content")]
        self.last_context = self._build_context_payload(context_results)
        analysis = self._fallback_analysis(base_risk)
        if self.model is not None:
            try:
                analysis = self._llm_case_analysis(data, context, base_risk)
            except Exception:
                analysis = self._fallback_analysis(base_risk)
        llm_risk = self._parse_risk(analysis.get("suggested_risk_level"))
        final_risk = self._max_risk(base_risk, llm_risk)
        reasons = [f"Triage support only. Rule-based risk screen: {base_risk.value}."]
        if self.last_context["retrieved_count"]:
            reasons.append(
                f"Retrieved {self.last_context['retrieved_count']} chunks from {', '.join(self.last_context['source_names']) or 'no named sources'}."
            )
            if self.last_context["summary"]:
                reasons.append("Context summary: " + self.last_context["summary"])
        if analysis.get("possible_concerns"):
            reasons.append("Possible concerns: " + ", ".join(analysis["possible_concerns"]))
        if analysis.get("doctor_summary"):
            reasons.append("Doctor summary: " + analysis["doctor_summary"])
        if analysis.get("confidence_note"):
            reasons.append(analysis["confidence_note"])
        return MediatorDecision(
            risk_level=final_risk,
            reasoning=" ".join(piece.strip() for piece in reasons if piece).strip(),
            recovery_plan=self.create_recovery_plan(final_risk),
            requires_doctor_communication=final_risk in {RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL},
        )

    def retrieve_relevant_context(self, patient_data: StructuredPatientData) -> list[str]:
        query = self._combined_text(patient_data)
        results = self._retrieve_context_results(query, k=3)
        return [item.get("content", "").strip() for item in results if item.get("content")] or [
            "General post-operative monitoring guidance is being used because no matching context was found."
        ]

    def create_recovery_plan(self, risk_level: RiskLevel) -> RecoveryPlan:
        steps_by_risk = {
            RiskLevel.LOW: [
                "Continue routine post-operative care as instructed.",
                "Take medications only as prescribed.",
                "Monitor warning signs.",
                "Attend scheduled follow-ups.",
            ],
            RiskLevel.MODERATE: [
                "Doctor review is required before final instructions.",
                "Monitor symptoms closely.",
                "Do not change medications without doctor advice.",
                "Seek urgent or emergency care if symptoms worsen.",
            ],
            RiskLevel.HIGH: [
                "Doctor review is required before final instructions.",
                "Monitor symptoms closely.",
                "Do not change medications without doctor advice.",
                "Seek urgent or emergency care if symptoms worsen.",
            ],
            RiskLevel.CRITICAL: [
                "Seek emergency medical care immediately.",
                "Do not wait for routine follow-up.",
                "Contact emergency services or go to nearest emergency room now.",
            ],
        }
        return RecoveryPlan(
            steps=[RecoveryPlanStep(step_number=index + 1, instruction=text) for index, text in enumerate(steps_by_risk[risk_level])]
        )

    def process_doctor_feedback(self, current_decision: MediatorDecision, feedback) -> MediatorDecision:
        if current_decision is None:
            raise ValueError("No current decision available to update.")
        risk = current_decision.risk_level
        note = (feedback.instructions or "").lower()
        if feedback.risk_override:
            downgrade_allowed = (
                risk != RiskLevel.CRITICAL
                or feedback.risk_override == RiskLevel.CRITICAL
                or "reviewed and stable" in note
                or "downgrade reason" in note
            )
            if downgrade_allowed:
                risk = feedback.risk_override
        plan = feedback.adjusted_recovery_plan or self._doctor_reviewed_plan(risk)
        reasoning = current_decision.reasoning
        if feedback.instructions:
            steps = list(plan.steps)
            steps.append(RecoveryPlanStep(step_number=len(steps) + 1, instruction=feedback.instructions))
            plan = RecoveryPlan(steps=steps)
            reasoning += " Doctor instructions were added to the final plan."
        if feedback.risk_override:
            if risk == feedback.risk_override:
                reasoning += " Doctor risk override applied."
            else:
                reasoning += " Critical risk was retained because no stability justification was provided."
        return MediatorDecision(
            risk_level=risk,
            reasoning=reasoning.strip(),
            recovery_plan=plan,
            requires_doctor_communication=False,
        )

    def process_patient_feedback(self, current_decision: MediatorDecision, feedback) -> MediatorDecision:
        if current_decision is None:
            raise ValueError("No current decision available to update.")
        pieces = [feedback.new_symptoms or "", feedback.notes_for_modification or ""]
        pieces.extend(question for question in feedback.questions if question)
        combined_feedback = " ".join(piece.strip() for piece in pieces if piece and piece.strip()).strip()
        if not combined_feedback and feedback.confirmed:
            return current_decision
        if not combined_feedback:
            return current_decision
        temp = StructuredPatientData(
            patient_info={},
            surgery_protocol=[],
            parsed_symptoms=[combined_feedback],
            medications_list=[],
            images_descriptions=[],
            documents_descriptions=[],
        )
        feedback_risk = self._screen_risk(self._combined_text(temp))
        context_results = self._retrieve_context_results(combined_feedback, k=3)
        self.last_context = self._build_context_payload(context_results)
        if feedback_risk == RiskLevel.LOW:
            feedback_risk = RiskLevel.MODERATE
        final_risk = self._max_risk(current_decision.risk_level, feedback_risk)
        return MediatorDecision(
            risk_level=final_risk,
            reasoning=f"{current_decision.reasoning} Patient feedback triggered re-triage based on: {combined_feedback}",
            recovery_plan=self.create_recovery_plan(final_risk),
            requires_doctor_communication=final_risk in {RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL},
        )

    def get_last_context(self):
        return {
            "retrieved_count": self.last_context.get("retrieved_count", 0),
            "source_names": list(self.last_context.get("source_names", [])),
            "summary": self.last_context.get("summary", ""),
            "retrieved_context": list(self.last_context.get("retrieved_context", [])),
            "sources": [dict(item) for item in self.last_context.get("sources", [])],
        }

    def _retrieve_context_results(self, query: str, k=3):
        if self.rag is None:
            return []
        return self.rag.search(query, k=k)

    def _build_context_payload(self, results):
        retrieved_context = [item.get("content", "").strip() for item in results if item.get("content")]
        source_names = list(dict.fromkeys(item.get("source", "unknown") for item in results if item.get("source")))
        sources = [
            {
                "source": item.get("source", "unknown"),
                "score": item.get("score"),
                "content_preview": self._preview(item.get("content", "")),
            }
            for item in results
        ]
        summary = " ".join(retrieved_context[:2]).strip()
        return {
            "retrieved_count": len(results),
            "source_names": source_names,
            "summary": summary,
            "retrieved_context": retrieved_context,
            "sources": sources,
        }

    def _combined_text(self, patient_data: StructuredPatientData) -> str:
        parts = []
        parts.extend(patient_data.parsed_symptoms)
        parts.extend(patient_data.surgery_protocol)
        parts.extend(patient_data.medications_list)
        parts.extend(patient_data.images_descriptions)
        parts.extend(patient_data.documents_descriptions)
        return " ".join(part for part in parts if part).lower()

    def _screen_risk(self, text: str) -> RiskLevel:
        if self._matches(text, self.CRITICAL_TERMS):
            return RiskLevel.CRITICAL
        if self._matches(text, self.HIGH_TERMS):
            return RiskLevel.HIGH
        if self._matches(text, self.MODERATE_TERMS):
            return RiskLevel.MODERATE
        return RiskLevel.LOW

    def _matches(self, text: str, terms: list[str]) -> bool:
        return any(self._term_found(text, term) for term in terms)

    def _term_found(self, text: str, term: str) -> bool:
        pattern = re.escape(term.lower())
        for match in re.finditer(pattern, text):
            prefix = text[max(0, match.start() - 60):match.start()].strip()
            if re.search(r"(no|not|without|denies|denied|watch for|monitor for|seek care if|if you have|لا يوجد|بدون|ليس)\s*$", prefix):
                continue
            return True
        return False

    def _llm_case_analysis(self, patient_data, context, base_risk):
        prompt = (
            "You are assisting with post-operative triage support only. Do not diagnose, prescribe, invent facts, or "
            "downgrade emergency signs. Return valid JSON with keys suggested_risk_level, possible_concerns, "
            "missing_information, clarifying_questions, doctor_summary, patient_explanation, confidence_note.\n"
            f"Base risk: {base_risk.value}\n"
            f"Context: {json.dumps(context, ensure_ascii=False)}\n"
            f"Patient data: {patient_data.model_dump_json()}"
        )
        return json.loads(self._json_text(self.model.invoke(prompt)))

    def _fallback_analysis(self, base_risk: RiskLevel) -> dict:
        return {
            "suggested_risk_level": base_risk.value,
            "possible_concerns": [],
            "missing_information": [],
            "clarifying_questions": [],
            "doctor_summary": "",
            "patient_explanation": "",
            "confidence_note": "Deterministic fallback analysis used.",
        }

    def _doctor_reviewed_plan(self, risk_level: RiskLevel) -> RecoveryPlan:
        if risk_level == RiskLevel.MODERATE:
            steps = [
                "Follow the reviewed plan provided by your care team.",
                "Monitor symptoms closely.",
                "Do not change medications without doctor advice.",
                "Seek urgent or emergency care if symptoms worsen.",
            ]
            return RecoveryPlan(steps=[RecoveryPlanStep(step_number=index + 1, instruction=text) for index, text in enumerate(steps)])
        if risk_level == RiskLevel.HIGH:
            steps = [
                "Follow the urgent reviewed plan provided by your care team.",
                "Monitor symptoms closely.",
                "Do not change medications without doctor advice.",
                "Seek urgent or emergency care if symptoms worsen.",
            ]
            return RecoveryPlan(steps=[RecoveryPlanStep(step_number=index + 1, instruction=text) for index, text in enumerate(steps)])
        return self.create_recovery_plan(risk_level)

    def _parse_risk(self, value):
        if not value:
            return None
        for risk in RiskLevel:
            if risk.value.lower() == str(value).lower():
                return risk
        return None

    def _max_risk(self, left: RiskLevel, right):
        if not right or left == RiskLevel.CRITICAL:
            return left
        return right if self.RANK[right] > self.RANK[left] else left

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

    def _preview(self, text, limit=180):
        compact = " ".join((text or "").split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    def _empty_context(self):
        return {
            "retrieved_count": 0,
            "source_names": [],
            "summary": "",
            "retrieved_context": [],
            "sources": [],
        }

    def _text(self, response):
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content).strip()

    def _json_text(self, response):
        text = self._text(response)
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        return text.strip()
