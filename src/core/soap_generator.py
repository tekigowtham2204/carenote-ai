"""
SOAP Note Generator — The Core Clinical Pipeline

This is where the product thinking lives.

Architecture: Transcript → Extraction → SOAP Draft → HITL Review → Final Note
Each stage is a separate LLM call with dedicated prompts and guardrails.

Design Decision: Why separate LLM calls per stage?
- Each stage has different failure modes
- Extraction errors compound if mixed with note generation
- Individual stages are independently evaluable (via CareNote Eval)
- HITL checkpoints slot cleanly between stages
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional
from ..llm.base import BaseLLMClient, LLMResponse
from ..prompts.soap_prompts import (
    TRANSCRIPT_EXTRACTION_PROMPT,
    SOAP_NOTE_PROMPT,
    BILLING_CODE_PROMPT,
)


@dataclass
class UncertaintyZone:
    """An area in the note where the AI isn't confident."""
    field: str
    content: str
    confidence: float
    reason: str

    @property
    def is_critical(self) -> bool:
        """Critical if it involves medication, diagnosis, or allergies."""
        critical_keywords = ["medication", "diagnosis", "dosage", "allergy", "drug"]
        return any(kw in self.field.lower() or kw in self.content.lower() for kw in critical_keywords)


@dataclass
class ExtractedData:
    """Structured clinical data extracted from a transcript."""
    chief_complaint: str = ""
    subjective: dict = field(default_factory=dict)
    objective: dict = field(default_factory=dict)
    assessment_hints: list = field(default_factory=list)
    plan_hints: list = field(default_factory=list)
    uncertainty_zones: list = field(default_factory=list)
    raw_json: dict = field(default_factory=dict)


@dataclass
class SOAPNote:
    """A complete SOAP note with metadata."""
    subjective: str = ""
    objective: str = ""
    assessment: str = ""
    plan: str = ""
    uncertainty_zones: list = field(default_factory=list)
    billing_codes: dict = field(default_factory=dict)
    full_text: str = ""
    generation_confidence: float = 0.0
    requires_human_review: bool = True  # Default to requiring review — safety first


class SOAPGenerator:
    """
    Multi-stage SOAP note generation pipeline.

    Stage 1: Extract structured data from transcript
    Stage 2: Generate SOAP note draft from extracted data
    Stage 3: Suggest billing codes
    """

    def __init__(self, llm_client: BaseLLMClient, uncertainty_threshold: float = 0.7):
        self.llm = llm_client
        self.uncertainty_threshold = uncertainty_threshold

    def extract_clinical_data(self, transcript: str) -> ExtractedData:
        """Stage 1: Extract structured clinical information from transcript."""
        prompt = TRANSCRIPT_EXTRACTION_PROMPT.format(transcript=transcript)

        response = self.llm.generate(
            prompt=prompt,
            system_prompt="You are a clinical data extraction specialist. Be precise. Flag uncertainties.",
            temperature=0.2,  # Even lower temp for extraction — precision matters
        )

        return self._parse_extracted_data(response.content)

    def generate_soap_note(self, extracted_data: ExtractedData) -> SOAPNote:
        """Stage 2: Generate SOAP note from extracted clinical data."""
        data_str = json.dumps(extracted_data.raw_json, indent=2) if extracted_data.raw_json else str(extracted_data)

        prompt = SOAP_NOTE_PROMPT.format(extracted_data=data_str)

        response = self.llm.generate(
            prompt=prompt,
            system_prompt="You are a clinical documentation specialist. Safety and accuracy above all.",
            temperature=0.3,
        )

        soap_note = self._parse_soap_note(response.content)
        soap_note.generation_confidence = response.confidence

        # If confidence is below threshold, force human review
        soap_note.requires_human_review = (
            response.confidence < self.uncertainty_threshold
            or len(soap_note.uncertainty_zones) > 0
            or "[REQUIRES VERIFICATION]" in response.content
        )

        return soap_note

    def suggest_billing_codes(self, assessment: str, plan: str) -> dict:
        """Stage 3: Suggest ICD-10 and CPT billing codes."""
        prompt = BILLING_CODE_PROMPT.format(assessment=assessment, plan=plan)

        response = self.llm.generate(
            prompt=prompt,
            system_prompt="You are a medical coding assistant. Suggest codes conservatively.",
            temperature=0.2,
        )

        return self._parse_billing_codes(response.content)

    def run_full_pipeline(self, transcript: str) -> SOAPNote:
        """Run the complete pipeline: Extract → Generate → Bill."""
        # Stage 1
        extracted = self.extract_clinical_data(transcript)

        # Stage 2
        soap_note = self.generate_soap_note(extracted)

        # Stage 3
        if soap_note.assessment:
            soap_note.billing_codes = self.suggest_billing_codes(
                soap_note.assessment, soap_note.plan
            )

        return soap_note

    def _parse_extracted_data(self, raw_content: str) -> ExtractedData:
        """Parse LLM response into structured ExtractedData."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{[\s\S]*\}', raw_content)
            if json_match:
                data = json.loads(json_match.group())
                return ExtractedData(
                    chief_complaint=data.get("chief_complaint", ""),
                    subjective=data.get("subjective", {}),
                    objective=data.get("objective", {}),
                    assessment_hints=data.get("assessment_hints", []),
                    plan_hints=data.get("plan_hints", []),
                    uncertainty_zones=data.get("uncertainty_zones", []),
                    raw_json=data,
                )
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: return raw content as chief complaint
        return ExtractedData(
            chief_complaint=raw_content[:500],
            uncertainty_zones=["Full extraction failed — manual review required"],
            raw_json={"raw_content": raw_content},
        )

    def _parse_soap_note(self, raw_content: str) -> SOAPNote:
        """Parse LLM response into structured SOAPNote."""
        sections = {
            "subjective": "",
            "objective": "",
            "assessment": "",
            "plan": "",
        }

        current_section = None
        uncertainty_zones = []

        for line in raw_content.split("\n"):
            line_lower = line.strip().lower()

            if "subjective" in line_lower and line.strip().startswith("#"):
                current_section = "subjective"
            elif "objective" in line_lower and line.strip().startswith("#"):
                current_section = "objective"
            elif "assessment" in line_lower and line.strip().startswith("#"):
                current_section = "assessment"
            elif "plan" in line_lower and line.strip().startswith("#"):
                current_section = "plan"
            elif "uncertainty" in line_lower and line.strip().startswith("#"):
                current_section = "uncertainty"
            elif "billing" in line_lower and line.strip().startswith("#"):
                current_section = "billing"
            elif current_section and current_section in sections:
                sections[current_section] += line + "\n"
            elif current_section == "uncertainty":
                if line.strip() and not line.strip().startswith("#"):
                    uncertainty_zones.append(
                        UncertaintyZone(
                            field="general",
                            content=line.strip(),
                            confidence=0.5,
                            reason="Flagged by LLM",
                        )
                    )

        return SOAPNote(
            subjective=sections["subjective"].strip(),
            objective=sections["objective"].strip(),
            assessment=sections["assessment"].strip(),
            plan=sections["plan"].strip(),
            uncertainty_zones=uncertainty_zones,
            full_text=raw_content,
            requires_human_review=True,
        )

    def _parse_billing_codes(self, raw_content: str) -> dict:
        """Parse billing code suggestions from LLM response."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', raw_content)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return {"raw_response": raw_content, "parse_error": True}
