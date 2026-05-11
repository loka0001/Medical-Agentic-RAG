import re

try:
    from agents.interaction_agent import InteractionAgent
except Exception:
    InteractionAgent = None

try:
    from agents.patient_mentor_agent import PatientMentorAgent
except Exception:
    PatientMentorAgent = None

from agents.mediator_agent import MediatorAgent
from schemas import MediatorDecision, MentorState, RiskLevel, StructuredPatientData
from services import PatientMessageService, RagService, Storage


class SystemOrchestrator:
    def __init__(self):
        self.rag = RagService()
        self.storage = Storage()
        self.interaction_agent = self._init_interaction_agent()
        self.patient_mentor_agent = self._init_patient_mentor()
        self.patient_messages = PatientMessageService()
        self.mediator_agent = MediatorAgent(rag_service=self.rag)
        self.pending_decisions: dict[str, MediatorDecision] = {}
        self.last_decisions: dict[str, MediatorDecision] = {}
        self.mentor_states: dict[str, MentorState] = {}
        self.last_patient_messages: dict[str, str] = {}
        self.latest_case_inputs: dict[str, dict] = {}
        self.decision_contexts: dict[str, dict] = {}

    def process_raw_input(self, patient_input):
        patient_id = self._patient_id(patient_input.patient_info)
        normalized = patient_input.model_copy(deep=True)
        normalized.patient_info = dict(normalized.patient_info)
        normalized.patient_info["patient_id"] = patient_id
        self.latest_case_inputs[patient_id] = normalized.model_dump()
        self.storage.save_case(patient_id, normalized)
        structured = self._process_with_interaction_agent(normalized)
        decision = self.mediator_agent.process_structured_data(structured)
        self.decision_contexts[patient_id] = self.mediator_agent.get_last_context()
        self.last_decisions[patient_id] = decision
        return self._handle_decision(patient_id, decision, origin="patient_submit")

    def process_doctor_feedback(self, feedback):
        patient_id = feedback.patient_id if feedback.patient_id in self.pending_decisions else self._latest_pending_id()
        if not patient_id:
            return self._error_response("No pending decision found for doctor feedback.")
        updated = self.mediator_agent.process_doctor_feedback(self.pending_decisions[patient_id], feedback)
        self.pending_decisions.pop(patient_id, None)
        self.last_decisions[patient_id] = updated
        self.storage.save_feedback(patient_id, "doctor", feedback)
        self.storage.save_decision(patient_id, "plan_delivered", updated)
        message = self.patient_messages.generate_patient_message(updated)
        mentor_state = self._build_mentor_state(updated.recovery_plan)
        self.last_patient_messages[patient_id] = message
        self.mentor_states[patient_id] = mentor_state
        return self._response_payload(
            status="plan_delivered",
            next_action="patient_follow_up",
            patient_id=patient_id,
            decision=updated,
            patient_message=message,
            case_state=self._case_state(patient_id, pending=False, mentor_state=mentor_state),
            context=self.decision_contexts.get(patient_id, self._empty_context()),
        )

    def process_patient_feedback(self, feedback):
        patient_id = feedback.patient_id or self._latest_known_id()
        current = self.pending_decisions.get(patient_id) or self.last_decisions.get(patient_id)
        if not patient_id or current is None:
            return self._error_response("No prior patient decision found for feedback.")
        updated = self.mediator_agent.process_patient_feedback(current, feedback)
        self.decision_contexts[patient_id] = self.mediator_agent.get_last_context()
        self.last_decisions[patient_id] = updated
        self.storage.save_feedback(patient_id, "patient", feedback)
        return self._handle_decision(patient_id, updated, origin="patient_feedback")

    def ingest_text(self, text, source="manual"):
        result = self.rag.ingest_text(text, source=source)
        self.storage.save_document(source, metadata={"text": text, **result})
        return result

    def ingest_file(self, file_path, source=None):
        result = self.rag.ingest_file(file_path, source=source)
        self.storage.save_document(result["source"], path=str(file_path), metadata=result)
        return result

    def search_knowledge(self, query):
        return {"query": query, "results": self.rag.search(query), "mode": self.rag.mode}

    def answer_from_knowledge(self, query: str, audience: str = "doctor"):
        return self.rag.answer(query, audience=audience)

    def get_patient_history(self, patient_id):
        return self.storage.get_patient_history(patient_id)

    def state(self):
        rag_health = self.rag.health()
        return {
            "pending_patient_ids": list(self.pending_decisions.keys()),
            "pending_case_summaries": [self._pending_case_summary(patient_id) for patient_id in self.pending_decisions],
            "last_decision_patient_ids": list(self.last_decisions.keys()),
            "mentor_states": {patient_id: state.current_step_index for patient_id, state in self.mentor_states.items()},
            "last_patient_messages": self.last_patient_messages,
            "rag_mode": rag_health["mode"],
            "rag_last_error": rag_health["last_error"],
            "rag_health": rag_health,
            "storage_summary": self.storage.state_summary(),
            "documents": self.rag.list_documents(),
            "stored_documents": self.storage.list_documents(),
        }

    def _handle_decision(self, patient_id, decision, origin):
        self.last_decisions[patient_id] = decision
        context = self.decision_contexts.get(patient_id, self._empty_context())
        if decision.risk_level == RiskLevel.CRITICAL:
            self.pending_decisions.pop(patient_id, None)
            self.storage.save_decision(patient_id, "emergency_escalation", decision)
            message = self.patient_messages.generate_patient_message(decision)
            self.last_patient_messages[patient_id] = message
            return self._response_payload(
                status="emergency_escalation",
                next_action="emergency_care",
                patient_id=patient_id,
                decision=decision,
                patient_message=message,
                case_state=self._case_state(patient_id, pending=False, mentor_state=self.mentor_states.get(patient_id)),
                context=context,
            )
        if decision.risk_level in {RiskLevel.MODERATE, RiskLevel.HIGH}:
            self.pending_decisions[patient_id] = decision
            self.storage.save_decision(patient_id, "doctor_review_required", decision)
            message = self.patient_messages.generate_patient_message(decision)
            self.last_patient_messages[patient_id] = message
            return self._response_payload(
                status="doctor_review_required",
                next_action="doctor_review",
                patient_id=patient_id,
                decision=decision,
                patient_message=message,
                case_state=self._case_state(patient_id, pending=True, mentor_state=self.mentor_states.get(patient_id)),
                context=context,
            )
        self.pending_decisions.pop(patient_id, None)
        self.storage.save_decision(patient_id, "plan_delivered", decision)
        message = self.patient_messages.generate_patient_message(decision)
        mentor_state = self._build_mentor_state(decision.recovery_plan)
        self.last_patient_messages[patient_id] = message
        self.mentor_states[patient_id] = mentor_state
        status = "updated_after_patient_feedback" if origin == "patient_feedback" else "plan_delivered"
        next_action = "continue_current_plan" if origin == "patient_feedback" else "patient_follow_up"
        return self._response_payload(
            status=status,
            next_action=next_action,
            patient_id=patient_id,
            decision=decision,
            patient_message=message,
            case_state=self._case_state(patient_id, pending=False, mentor_state=mentor_state),
            context=context,
        )

    def _response_payload(self, status, next_action, patient_id, decision, patient_message, case_state, context):
        return {
            "status": status,
            "next_action": next_action,
            "patient_id": patient_id,
            "risk_level": decision.risk_level,
            "doctor_review_required": status == "doctor_review_required",
            "patient_message": patient_message,
            "recovery_plan": decision.recovery_plan,
            "reasoning": decision.reasoning,
            "risk_reason": decision.reasoning,
            "retrieved_context_summary": context.get("summary", ""),
            "retrieved_context": context.get("retrieved_context", []),
            "sources": context.get("sources", []),
            "case_state": case_state,
            "decision": decision,
        }

    def _pending_case_summary(self, patient_id):
        decision = self.pending_decisions.get(patient_id)
        case = self.latest_case_inputs.get(patient_id, {})
        context = self.decision_contexts.get(patient_id, self._empty_context())
        return {
            "patient_id": patient_id,
            "age": (case.get("patient_info") or {}).get("age"),
            "symptoms": case.get("symptoms"),
            "surgery_protocol": case.get("surgery_protocol"),
            "risk_level": decision.risk_level.value if decision else None,
            "reasoning": decision.reasoning if decision else "",
            "recovery_plan": decision.recovery_plan if decision else None,
            "retrieved_context_summary": context.get("summary", ""),
            "sources": context.get("sources", []),
        }

    def _case_state(self, patient_id, pending, mentor_state):
        return {
            "pending": pending,
            "mentor_step": mentor_state.current_step_index if mentor_state else None,
            "pending_queue_size": len(self.pending_decisions),
            "rag_mode": self.rag.health()["mode"],
            "has_patient_message": patient_id in self.last_patient_messages,
        }

    def _error_response(self, message):
        return {
            "status": "error",
            "next_action": "no_action_required",
            "patient_id": None,
            "risk_level": None,
            "doctor_review_required": False,
            "patient_message": None,
            "recovery_plan": None,
            "reasoning": message,
            "risk_reason": message,
            "retrieved_context_summary": "",
            "retrieved_context": [],
            "sources": [],
            "case_state": {},
            "message": message,
        }

    def _init_interaction_agent(self):
        if InteractionAgent is None:
            return None
        try:
            return InteractionAgent()
        except Exception:
            return None

    def _init_patient_mentor(self):
        if PatientMentorAgent is None:
            return None
        try:
            return PatientMentorAgent()
        except Exception:
            return None

    def _process_with_interaction_agent(self, patient_input):
        if self.interaction_agent is not None:
            try:
                return self.interaction_agent.process_raw_input(patient_input)
            except Exception:
                pass
        return self._fallback_parse(patient_input)

    def _build_mentor_state(self, recovery_plan):
        if self.patient_mentor_agent is not None:
            try:
                return self.patient_mentor_agent.process_confirmed_plan(recovery_plan)
            except Exception:
                pass
        return MentorState(recovery_plan=recovery_plan, current_step_index=0)

    def _fallback_parse(self, patient_input):
        return StructuredPatientData(
            patient_info=dict(patient_input.patient_info),
            surgery_protocol=self._split_items(patient_input.surgery_protocol),
            parsed_symptoms=self._split_items(patient_input.symptoms),
            medications_list=self._split_items(patient_input.medications),
            images_descriptions=[item.description.strip() for item in patient_input.images or [] if item.description],
            documents_descriptions=[item.description.strip() for item in patient_input.documents or [] if item.description],
        )

    def _split_items(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            parts = [piece for entry in value for piece in self._split_items(entry)]
            return list(dict.fromkeys(parts))
        text = re.sub(r"\b(?:and)\b", "\n", str(value), flags=re.IGNORECASE)
        text = re.sub(r"\sو\s", "\n", text.replace("\r", "\n"))
        text = re.sub(r"[;,/]", "\n", text)
        return list(dict.fromkeys(item.strip(" -\t") for item in text.split("\n") if item.strip(" -\t")))

    def _patient_id(self, patient_info):
        patient_id = str((patient_info or {}).get("patient_id") or "").strip()
        return patient_id or f"patient-{len(self.last_decisions) + len(self.pending_decisions) + 1}"

    def _latest_pending_id(self):
        return next(reversed(self.pending_decisions), None) if self.pending_decisions else None

    def _latest_known_id(self):
        for mapping in (self.pending_decisions, self.last_decisions):
            if mapping:
                return next(reversed(mapping))
        return None

    def _empty_context(self):
        return {
            "retrieved_count": 0,
            "source_names": [],
            "summary": "",
            "retrieved_context": [],
            "sources": [],
        }
