"""
Google Gemini LLM Client

Same interface, different vendor. That's the whole point.
"""

import google.generativeai as genai
from .base import BaseLLMClient, LLMResponse


class GeminiClient(BaseLLMClient):
    """Google Gemini backend client."""

    def __init__(self, api_key: str, model: str = "gemini-pro"):
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.3) -> LLMResponse:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        try:
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature),
            )
            return LLMResponse(
                content=response.text,
                model=self.model_name,
                confidence=1.0,
            )
        except Exception as e:
            return LLMResponse(
                content=f"[ERROR] Gemini call failed: {str(e)}",
                model=self.model_name,
                confidence=0.0,
            )

    def is_available(self) -> bool:
        try:
            self.model.generate_content("ping")
            return True
        except Exception:
            return False
