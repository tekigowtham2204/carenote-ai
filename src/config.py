"""
Configuration module — Multi-LLM Backend Support

Design Decision: Vendor-agnostic LLM configuration.
A production PM designs for optionality, not vendor lock-in.
Supports OpenAI, Google Gemini, and Ollama (local) out of the box.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """LLM backend configuration loaded from environment variables."""

    backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "openai"))

    # OpenAI
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4"))

    # Gemini
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-pro"))

    # Ollama
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3"))

    # App
    debug: bool = field(
        default_factory=lambda: os.getenv("APP_DEBUG", "false").lower() == "true"
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def validate(self) -> bool:
        """Validate that the selected backend has the required credentials."""
        if self.backend == "openai" and not self.openai_api_key:
            return False
        if self.backend == "gemini" and not self.gemini_api_key:
            return False
        # Ollama doesn't need API keys (runs locally)
        return True


@dataclass
class AppConfig:
    """Application-level configuration."""

    app_name: str = "CareNote AI"
    app_description: str = "Human-in-the-Loop Clinical Documentation Copilot"
    version: str = "0.1.0"

    # HITL thresholds — these are product decisions, not engineering decisions
    # A PM defines what "uncertain enough to require human review" means
    uncertainty_threshold: float = 0.7  # Below this → flag for human review
    critical_fields: list = field(
        default_factory=lambda: ["diagnosis", "medication", "dosage", "allergies"]
    )

    # Success metrics from the PRD (mirrors resume)
    target_time_reduction: float = 0.35  # ≥35% time reduction
    target_nps: int = 40                  # NPS ≥40
    target_accuracy_f1: float = 0.87      # Note accuracy F1 ≥0.87

    llm: LLMConfig = field(default_factory=LLMConfig)


def get_config() -> AppConfig:
    """Factory function to get the app configuration."""
    return AppConfig()
