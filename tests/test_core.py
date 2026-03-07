"""
Tests for CareNote AI core logic.

These tests validate core business logic without requiring LLM API calls.
"""

import pytest
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.audio_processor import AudioProcessor, TranscriptSegment, ProcessedTranscript
from src.core.soap_generator import SOAPGenerator, SOAPNote, ExtractedData, UncertaintyZone
from src.core.hitl_engine import HITLEngine, ReviewSession, Checkpoint, ReviewDecision, ReviewPriority
from src.llm.base import BaseLLMClient, LLMResponse


# ============================================================
# Mock LLM Client for testing
# ============================================================
class MockLLMClient(BaseLLMClient):
    def __init__(self, response_content="Mock response"):
        self.response_content = response_content
        self.call_count = 0

    def generate(self, prompt, system_prompt="", temperature=0.3):
        self.call_count += 1
        return LLMResponse(content=self.response_content, model="mock", confidence=0.85)

    def is_available(self):
        return True


# ============================================================
# Audio Processor Tests
# ============================================================
class TestAudioProcessor:
    def test_parse_simple_conversation(self):
        processor = AudioProcessor()
        transcript = "Doctor: What brings you in today?\nPatient: I have a headache."
        result = processor.process_text_input(transcript)
        assert len(result.segments) == 2
        assert result.segments[0].speaker == "doctor"
        assert result.segments[1].speaker == "patient"

    def test_parse_multiline_dialogue(self):
        processor = AudioProcessor()
        transcript = "Doctor: How are you feeling?\nThe symptoms started when?\nPatient: About a week ago."
        result = processor.process_text_input(transcript)
        assert len(result.segments) == 2
        assert "symptoms started" in result.segments[0].text

    def test_duration_estimate(self):
        processor = AudioProcessor()
        transcript = "Doctor: Hello.\nPatient: Hi."
        result = processor.process_text_input(transcript)
        assert result.duration_minutes > 0

    def test_empty_transcript(self):
        processor = AudioProcessor()
        result = processor.process_text_input("")
        assert len(result.segments) == 0


# ============================================================
# SOAP Generator Tests
# ============================================================
class TestSOAPGenerator:
    def test_extraction_pipeline_calls_llm(self):
        mock_response = json.dumps({
            "chief_complaint": "Headache",
            "subjective": {"symptoms": ["headache"]},
            "objective": {},
            "assessment_hints": [],
            "plan_hints": [],
            "uncertainty_zones": []
        })
        client = MockLLMClient(response_content=mock_response)
        generator = SOAPGenerator(llm_client=client)
        result = generator.extract_clinical_data("Doctor: Test\nPatient: Test")
        assert client.call_count == 1
        assert result.chief_complaint == "Headache"

    def test_soap_note_requires_review_by_default(self):
        note = SOAPNote()
        assert note.requires_human_review is True

    def test_uncertainty_zone_critical_detection(self):
        zone = UncertaintyZone(
            field="medication",
            content="Dosage unclear",
            confidence=0.4,
            reason="Low confidence"
        )
        assert zone.is_critical is True

    def test_uncertainty_zone_non_critical(self):
        zone = UncertaintyZone(
            field="follow_up_date",
            content="3 weeks vs 4 weeks",
            confidence=0.6,
            reason="Minor ambiguity"
        )
        assert zone.is_critical is False


# ============================================================
# HITL Engine Tests
# ============================================================
class TestHITLEngine:
    def test_create_review_session(self):
        engine = HITLEngine()
        note = SOAPNote(
            assessment="Tension headache",
            plan="Ibuprofen 400mg TID",
        )
        session = engine.create_review_session(note)
        assert isinstance(session, ReviewSession)
        assert session.total_checkpoints >= 2  # At least diagnosis + medication

    def test_critical_checkpoints_generated(self):
        engine = HITLEngine()
        note = SOAPNote(assessment="Test diagnosis", plan="Test medication")
        session = engine.create_review_session(note)
        urgent = [cp for cp in session.checkpoints if cp.priority == ReviewPriority.URGENT]
        assert len(urgent) >= 2

    def test_resolve_checkpoint(self):
        engine = HITLEngine()
        note = SOAPNote(assessment="Test", plan="Test")
        session = engine.create_review_session(note)
        cp_id = session.checkpoints[0].id
        resolved = engine.resolve_checkpoint(session, cp_id, ReviewDecision.APPROVED)
        assert resolved.decision == ReviewDecision.APPROVED
        assert resolved.reviewed_at is not None

    def test_session_completion(self):
        engine = HITLEngine()
        note = SOAPNote(assessment="Test", plan="Test")
        session = engine.create_review_session(note)
        for cp in session.checkpoints:
            engine.resolve_checkpoint(session, cp.id, ReviewDecision.APPROVED)
        assert session.is_complete is True
        assert session.completion_percentage == 100.0

    def test_audit_log_generation(self):
        engine = HITLEngine()
        note = SOAPNote(assessment="Test", plan="Test")
        session = engine.create_review_session(note)
        audit = engine.get_audit_log(session)
        assert "session_id" in audit
        assert "checkpoints" in audit
        assert audit["total_checkpoints"] == session.total_checkpoints

    def test_invalid_checkpoint_raises(self):
        engine = HITLEngine()
        note = SOAPNote(assessment="Test", plan="Test")
        session = engine.create_review_session(note)
        with pytest.raises(ValueError):
            engine.resolve_checkpoint(session, "nonexistent_id", ReviewDecision.APPROVED)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
