"""
OpenAI LLM Client

Production note: Temperature is kept low (0.3) by default for clinical use.
You do NOT want creative writing when generating medication dosages.
"""

from openai import OpenAI
from .base import BaseLLMClient, LLMResponse


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT backend client."""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.3) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
            )
        except Exception as e:
            return LLMResponse(
                content=f"[ERROR] OpenAI call failed: {str(e)}",
                model=self.model,
                confidence=0.0,
            )

    def is_available(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False
