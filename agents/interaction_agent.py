import json
import os
import re

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from schemas.patient_input import PatientInput
from schemas.structured_patient_data import StructuredPatientData

load_dotenv()


class InteractionAgent:
    """Agent responsible for parsing raw multi-modal inputs."""

    def __init__(self):
        self.model = self._build_model()
        self.system_prompt = """
You are a medical data structuring assistant.

Your task is to convert raw patient input into clean, structured data
that matches the StructuredPatientData schema.

Rules:
- Keep all provided patient facts.
- Preserve Arabic and English.
- Do not invent missing data.
- Return surgery_protocol, parsed_symptoms, medications_list, images_descriptions,
  and documents_descriptions as lists.
- Do not provide diagnosis, risk assessment, or medical advice.
- Return only structured data matching StructuredPatientData.
"""
        self.agent = None
        if self.model is not None:
            try:
                self.agent = create_agent(
                    model=self.model,
                    system_prompt=self.system_prompt,
                    checkpointer=InMemorySaver(),
                    response_format=ToolStrategy(StructuredPatientData),
                )
            except Exception:
                self.agent = None

    def process_raw_input(self, raw_input: PatientInput, thread_id="1") -> StructuredPatientData:
        if self.agent is not None:
            try:
                result = self.agent.invoke(
                    {"messages": self._prepare_messages(raw_input)},
                    {"configurable": {"thread_id": thread_id}},
                )
                structured = result.get("structured_response")
                if isinstance(structured, StructuredPatientData):
                    return structured
            except Exception:
                pass
        return self._fallback_parse(raw_input)

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

    def _prepare_messages(self, raw_input: PatientInput):
        file_context = []
        for image in raw_input.images or []:
            file_context.append(f"image_description: {self._safe_media_description(image)}")
        for doc in raw_input.documents or []:
            file_context.append(f"document_description: {self._safe_media_description(doc)}")
        payload = {
            "patient_info": raw_input.patient_info,
            "surgery_protocol": raw_input.surgery_protocol,
            "symptoms": raw_input.symptoms,
            "medications": raw_input.medications,
            "images": file_context[:10],
            "documents": file_context[10:20] if len(file_context) > 10 else [],
        }
        return [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]

    def _safe_media_description(self, item):
        description = (getattr(item, "description", "") or "").strip()
        if not description:
            return "No description provided."
        if not self.model or not getattr(item, "file_data", None):
            return description
        try:
            analyzed = self._analyze_file(item)
            return analyzed or description
        except Exception:
            return description

    def _analyze_file(self, item):
        prompt = (
            "Summarize this medical file reference in one short factual phrase using the provided description. "
            "Do not diagnose. Do not add details that are not present.\n"
            f"Description: {item.description}"
        )
        return self._text(self.model.invoke(prompt))

    def _fallback_parse(self, raw_input: PatientInput) -> StructuredPatientData:
        return StructuredPatientData(
            patient_info=dict(raw_input.patient_info),
            surgery_protocol=self._split_items(raw_input.surgery_protocol),
            parsed_symptoms=self._split_items(raw_input.symptoms),
            medications_list=self._split_items(raw_input.medications),
            images_descriptions=[(item.description or "").strip() for item in raw_input.images or [] if (item.description or "").strip()],
            documents_descriptions=[(item.description or "").strip() for item in raw_input.documents or [] if (item.description or "").strip()],
        )

    def _split_items(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            items = [piece for entry in value for piece in self._split_items(entry)]
            return list(dict.fromkeys(items))
        text = re.sub(r"\b(?:and)\b", "\n", str(value), flags=re.IGNORECASE)
        text = re.sub(r"\sو\s", "\n", text.replace("\r", "\n"))
        text = re.sub(r"[,;/]", "\n", text)
        items = [item.strip(" -\t") for item in text.split("\n") if item.strip(" -\t")]
        return list(dict.fromkeys(items))

    def _text(self, response):
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content).strip()
