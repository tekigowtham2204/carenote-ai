"""
Ollama LLM Client — Local/Open-Source Backend

For teams that can't send patient data to cloud APIs.
Runs entirely on your machine. Zero data leaves the building.
"""

import requests
from .base import BaseLLMClient, LLMResponse


class OllamaClient(BaseLLMClient):
    """Ollama local LLM backend client."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.3) -> LLMResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }

        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                content=data.get("response", ""),
                model=self.model,
                tokens_used=data.get("eval_count", 0),
                raw_response=data,
            )
        except Exception as e:
            return LLMResponse(
                content=f"[ERROR] Ollama call failed: {str(e)}",
                model=self.model,
                confidence=0.0,
            )

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
