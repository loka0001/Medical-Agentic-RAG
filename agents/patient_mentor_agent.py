import os
import re
from typing import List, Literal, Optional, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI

from schemas.mentor_state import MentorState
from schemas.patient_feedback import PatientFeedback
from schemas.recovery_plan import RecoveryPlan

load_dotenv()


class PatientUpdateDecision(BaseModel):
    action: Literal["completion", "question", "medical_review", "plan_modification"] = Field(
        description="How the patient update should be handled."
    )
    extracted_question: Optional[str] = Field(default=None)
    review_note: Optional[str] = Field(default=None)


class PatientMentorAgent:
    """Agent responsible for mentoring the patient and tracking progress."""

    def __init__(self):
        self.model = self._build_model()
        self.system_prompt = """
You are a patient update triage assistant.

Classify the patient's message into exactly one action:
- completion
- question
- medical_review
- plan_modification

Rules:
- If the message reports concerning symptoms, use medical_review.
- If the message asks a question without concerning symptoms, use question.
- If the message asks to change or stop the plan, use plan_modification.
- If the patient clearly completed the step, use completion.
- Do not give medical advice.
- Return only structured output matching the schema.
"""
        self.agent = None
        if self.model is not None:
            try:
                self.agent = create_agent(
                    model=self.model,
                    system_prompt=self.system_prompt,
                    response_format=ToolStrategy(PatientUpdateDecision),
                )
            except Exception:
                self.agent = None

    def process_confirmed_plan(self, plan: RecoveryPlan) -> MentorState:
        if not plan.steps:
            raise ValueError("Cannot process a recovery plan with no steps.")
        ordered_steps = sorted(plan.steps, key=lambda step: step.step_number)
        ordered_plan = RecoveryPlan(steps=ordered_steps)
        return MentorState(recovery_plan=ordered_plan, current_step_index=0)

    def process_patient_update(self, current_state: MentorState, patient_update: str) -> Union[MentorState, PatientFeedback]:
        text = patient_update.strip()
        if not text:
            return self._build_feedback("Empty update received.", [], None)
        if self.agent is not None:
            try:
                current_step_text = self._get_current_step_text(current_state)
                result = self.agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Current recovery step: {current_step_text}\nPatient message: {patient_update}",
                            }
                        ]
                    }
                )
                return self._handle_decision(current_state, patient_update, result["structured_response"])
            except Exception:
                pass
        return self._fallback_classify(current_state, patient_update)

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

    def _handle_decision(self, current_state, patient_update, decision):
        if decision.action == "completion":
            return self._advance_plan_step(current_state)
        if decision.action == "question":
            return self._build_feedback(None, [decision.extracted_question or patient_update], None)
        if decision.action == "plan_modification":
            return self._build_feedback(decision.review_note or patient_update, [], None)
        return self._build_feedback(decision.review_note or patient_update, [], decision.review_note or patient_update)

    def _fallback_classify(self, current_state, patient_update):
        text = patient_update.strip()
        lowered = text.lower()
        if self._contains_any(lowered, ["fever", "bleeding", "swelling", "pain", "pus", "redness", "chest pain", "difficulty breathing", "heavy bleeding", "cannot breathe", "حمى", "نزيف", "تورم", "ألم", "صديد", "احمرار", "صعوبة تنفس"]):
            return self._build_feedback(None, [], text)
        if self._contains_any(lowered, ["change", "stop", "skip", "adjust", "modify", "أغير", "أوقف", "أتخطى"]):
            return self._build_feedback(text, [], None)
        if "?" in text or re.match(r"^(how|when|what|why|can i|should i|هل|كيف|متى|ماذا)\b", lowered):
            return self._build_feedback(None, [text], None)
        if self._contains_any(lowered, ["done", "completed", "finished", "تم", "خلصت", "انتهيت"]):
            return self._advance_plan_step(current_state)
        return self._build_feedback(None, [], text)

    def _contains_any(self, text, terms):
        return any(term in text for term in terms)

    def _get_current_step_text(self, state: MentorState) -> str:
        steps = state.recovery_plan.steps
        if not steps:
            return "No current step."
        if state.current_step_index >= len(steps):
            return "Plan already completed."
        step = steps[state.current_step_index]
        return f"Step {step.step_number}: {step.instruction}"

    def _advance_plan_step(self, state: MentorState) -> MentorState:
        total_steps = len(state.recovery_plan.steps)
        next_index = state.current_step_index + 1
        if next_index >= total_steps:
            next_index = total_steps - 1
        return state.model_copy(update={"current_step_index": next_index})

    def _build_feedback(self, note: Optional[str], questions: List[str], new_symptoms: Optional[str]) -> PatientFeedback:
        return PatientFeedback(
            confirmed=False,
            notes_for_modification=note,
            questions=questions,
            new_symptoms=new_symptoms,
        )
