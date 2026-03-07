"""
Abstract LLM Interface

Design Decision: Every LLM backend implements the same contract.
This is not over-engineering — it's the difference between a demo
and a system that survives a vendor pricing change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Standardized response from any LLM backend."""
    content: str
    model: str
    tokens_used: int = 0
    confidence: float = 1.0
    raw_response: Optional[dict] = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.3) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: The user/task prompt
            system_prompt: System-level instructions
            temperature: Controls randomness (low = more deterministic for clinical use)

        Returns:
            LLMResponse with standardized fields
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is reachable and configured."""
        pass
