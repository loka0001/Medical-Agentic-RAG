import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from schemas import RiskLevel

load_dotenv()


class PatientMessageService:
    FALLBACKS = {
        RiskLevel.LOW: "Based on the information provided, your update appears low risk. Continue routine post-operative care as instructed. Watch for fever, redness, discharge, severe pain, bleeding, chest pain, or difficulty breathing.",
        RiskLevel.MODERATE: "Your update requires doctor review before final instructions can be given. Please monitor your symptoms and do not change medications without your doctor's guidance. Seek urgent care if symptoms worsen.",
        RiskLevel.HIGH: "Your update requires urgent doctor review. Monitor your symptoms closely and do not change medications without your doctor. Seek urgent care immediately if symptoms rapidly worsen.",
        RiskLevel.CRITICAL: "This update contains emergency warning signs. Seek emergency medical care immediately or contact local emergency services now. Do not wait for routine follow-up.",
    }
    REVIEWED_FALLBACKS = {
        RiskLevel.MODERATE: "Your doctor has reviewed this update. Follow the plan provided, monitor symptoms closely, and seek urgent care if symptoms worsen.",
        RiskLevel.HIGH: "Your doctor has reviewed this update. Follow the urgent follow-up plan provided and seek urgent care immediately if symptoms rapidly worsen.",
    }

    def __init__(self):
        self.model = self._build_model()

    def generate_patient_message(self, decision):
        fallback = self._fallback_message(decision)
        if self.model is None:
            return fallback
        try:
            prompt = (
                "Write a short patient-facing message for post-operative triage support. Do not diagnose, do not prescribe, "
                "and do not reduce urgency. If risk is Critical, clearly say to seek emergency care immediately. If doctor "
                "review is required, clearly say final instructions depend on doctor review.\n"
                f"Risk: {decision.risk_level.value}\n"
                f"Requires doctor communication: {decision.requires_doctor_communication}\n"
                f"Reasoning: {decision.reasoning}"
            )
            text = self._text(self.model.invoke(prompt))
            return text or fallback
        except Exception:
            return fallback

    def _fallback_message(self, decision):
        if not decision.requires_doctor_communication and decision.risk_level in self.REVIEWED_FALLBACKS:
            return self.REVIEWED_FALLBACKS[decision.risk_level]
        return self.FALLBACKS[decision.risk_level]

    def _build_model(self):
        if os.getenv("OPENAI_API_KEY"):
            try:
                return ChatOpenAI(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    api_key=os.getenv("OPENAI_API_KEY"),
                    temperature=0.1,
                )
            except Exception:
                return None
        if os.getenv("GITHUB_TOKEN"):
            try:
                return ChatOpenAI(
                    model=os.getenv("GITHUB_MODEL", "gpt-4o"),
                    api_key=os.getenv("GITHUB_TOKEN"),
                    base_url="https://models.inference.ai.azure.com",
                    temperature=0.1,
                )
            except Exception:
                return None
        return None

    def _text(self, response):
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content).strip()
