"""
Audio Processor — Simulated Ambient Voice → Text Pipeline

In production, this would integrate with ASR services (Whisper, Deepgram, etc.)
For the MVP, we simulate the pipeline with pre-built transcripts and a text input mode.

Why simulation is fine for a PM portfolio:
The value isn't in building another ASR wrapper — it's in designing the pipeline
architecture and the HITL checkpoints that come after.
"""

from dataclasses import dataclass
from typing import Optional
import json
import os


@dataclass
class TranscriptSegment:
    """A segment of a doctor-patient conversation."""
    speaker: str  # "doctor" or "patient"
    text: str
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0


@dataclass
class ProcessedTranscript:
    """Full processed transcript ready for SOAP generation."""
    segments: list
    full_text: str
    duration_minutes: float
    patient_name: str = "Anonymized Patient"
    encounter_date: str = ""
    encounter_type: str = "outpatient"


class AudioProcessor:
    """Processes ambient audio into structured transcripts.

    MVP Implementation: Works with text input and sample transcripts.
    Production would add: Whisper ASR, diarization, noise filtering.
    """

    def __init__(self, samples_dir: str = "data/samples"):
        self.samples_dir = samples_dir

    def process_text_input(self, raw_text: str) -> ProcessedTranscript:
        """Process raw text as a transcript (text-input mode)."""
        segments = self._parse_conversation(raw_text)
        return ProcessedTranscript(
            segments=segments,
            full_text=raw_text,
            duration_minutes=len(raw_text) / 500,  # Rough estimate
        )

    def load_sample_transcript(self, sample_name: str) -> ProcessedTranscript:
        """Load a pre-built sample transcript for demo purposes."""
        file_path = os.path.join(self.samples_dir, f"{sample_name}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Sample transcript '{sample_name}' not found at {file_path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        segments = [
            TranscriptSegment(
                speaker=seg["speaker"],
                text=seg["text"],
                timestamp_start=seg.get("start", 0),
                timestamp_end=seg.get("end", 0),
            )
            for seg in data.get("segments", [])
        ]

        return ProcessedTranscript(
            segments=segments,
            full_text=data.get("full_text", " ".join(s.text for s in segments)),
            duration_minutes=data.get("duration_minutes", 0),
            patient_name=data.get("patient_name", "Anonymized Patient"),
            encounter_date=data.get("encounter_date", ""),
            encounter_type=data.get("encounter_type", "outpatient"),
        )

    def get_available_samples(self) -> list:
        """List available sample transcripts."""
        if not os.path.exists(self.samples_dir):
            return []
        return [
            f.replace(".json", "")
            for f in os.listdir(self.samples_dir)
            if f.endswith(".json")
        ]

    def _parse_conversation(self, raw_text: str) -> list:
        """Parse raw text into conversation segments.

        Expected format:
        Doctor: text here
        Patient: text here
        """
        segments = []
        current_speaker = None
        current_text = []

        for line in raw_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.lower().startswith("doctor:"):
                if current_speaker and current_text:
                    segments.append(TranscriptSegment(
                        speaker=current_speaker,
                        text=" ".join(current_text),
                    ))
                current_speaker = "doctor"
                current_text = [line.split(":", 1)[1].strip()]
            elif line.lower().startswith("patient:"):
                if current_speaker and current_text:
                    segments.append(TranscriptSegment(
                        speaker=current_speaker,
                        text=" ".join(current_text),
                    ))
                current_speaker = "patient"
                current_text = [line.split(":", 1)[1].strip()]
            else:
                current_text.append(line)

        # Add the last segment
        if current_speaker and current_text:
            segments.append(TranscriptSegment(
                speaker=current_speaker,
                text=" ".join(current_text),
            ))

        return segments
