import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


class BaseAgent:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        return self._create_model(temperature=0)

    def _create_model(self, temperature: float):
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GITHUB_TOKEN")
        if not api_key:
            return None
        kwargs = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "api_key": api_key,
            "temperature": temperature,
        }
        if os.getenv("GITHUB_TOKEN") and not os.getenv("OPENAI_API_KEY"):
            kwargs["base_url"] = "https://models.inference.ai.azure.com"
            kwargs["model"] = os.getenv("GITHUB_MODEL", "gpt-4o")
        return ChatOpenAI(**kwargs)

    def _text(self, response) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content).strip()

    def _json_text(self, response) -> str:
        text = self._text(response)
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        return text.strip()
