"""
Human-in-the-Loop (HITL) Engine

This is the product decision that separates CareNote from every other AI scribe.

The thesis: AI scribes hallucinate clinical facts. The solution isn't better models —
it's better checkpoints. CareNote turns the scribe into a draft-and-verify tool
by forcing 1-click human verification at critical medical decision points.

Checkpoint placement is a PRODUCT decision, not an engineering decision:
- Diagnosis: Because a wrong diagnosis cascades into wrong treatment
- Medication: Because a hallucinated medication can harm a patient
- Dosage: Because 10mg vs 100mg is a decimal point but a life in practice
- Allergies: Because missing an allergy interaction is malpractice

A PM who understands where to put the checkpoints understands the domain.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from typing import Optional, Callable


class ReviewPriority(Enum):
    URGENT = "urgent"      # Safety-critical: medication conflicts, allergy issues
    STANDARD = "standard"  # Important: diagnosis verification, dosage confirmation
    LOW = "low"           # Informational: billing codes, formatting


class ReviewDecision(Enum):
    APPROVED = "approved"
    REVISED = "revised"
    ESCALATED = "escalated"
    PENDING = "pending"


@dataclass
class Checkpoint:
    """A single HITL checkpoint requiring human review."""
    id: str
    field_name: str
    original_value: str
    confidence: float
    priority: ReviewPriority
    reason: str
    decision: ReviewDecision = ReviewDecision.PENDING
    revised_value: Optional[str] = None
    reviewer_notes: str = ""
    reviewed_at: Optional[str] = None

    @property
    def is_resolved(self) -> bool:
        return self.decision != ReviewDecision.PENDING

    @property
    def is_critical(self) -> bool:
        return self.priority == ReviewPriority.URGENT


@dataclass
class ReviewSession:
    """A complete HITL review session with audit trail."""
    session_id: str
    checkpoints: list = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    reviewer_id: str = "physician_1"

    @property
    def total_checkpoints(self) -> int:
        return len(self.checkpoints)

    @property
    def resolved_count(self) -> int:
        return sum(1 for cp in self.checkpoints if cp.is_resolved)

    @property
    def pending_count(self) -> int:
        return self.total_checkpoints - self.resolved_count

    @property
    def critical_pending(self) -> list:
        return [cp for cp in self.checkpoints if cp.is_critical and not cp.is_resolved]

    @property
    def is_complete(self) -> bool:
        return all(cp.is_resolved for cp in self.checkpoints)

    @property
    def completion_percentage(self) -> float:
        if self.total_checkpoints == 0:
            return 100.0
        return (self.resolved_count / self.total_checkpoints) * 100


class HITLEngine:
    """
    Human-in-the-Loop verification engine for clinical documentation.

    Creates checkpoints at critical medical decision points and manages
    the review workflow with full audit trail.
    """

    def __init__(self, confidence_threshold: float = 0.7, critical_fields: list = None):
        self.confidence_threshold = confidence_threshold
        self.critical_fields = critical_fields or [
            "diagnosis", "medication", "dosage", "allergies"
        ]
        self._checkpoint_counter = 0

    def create_review_session(self, soap_note, uncertainty_zones: list = None) -> ReviewSession:
        """Create a review session with checkpoints from a SOAP note."""
        session = ReviewSession(session_id=f"review_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")

        # Auto-generate checkpoints from SOAP note
        checkpoints = self._generate_checkpoints(soap_note, uncertainty_zones or [])
        session.checkpoints = checkpoints

        return session

    def resolve_checkpoint(
        self,
        session: ReviewSession,
        checkpoint_id: str,
        decision: ReviewDecision,
        revised_value: str = None,
        reviewer_notes: str = "",
    ) -> Checkpoint:
        """Resolve a checkpoint with a doctor's decision."""
        for cp in session.checkpoints:
            if cp.id == checkpoint_id:
                cp.decision = decision
                cp.revised_value = revised_value
                cp.reviewer_notes = reviewer_notes
                cp.reviewed_at = datetime.now(timezone.utc).isoformat()

                # Check if session is complete
                if session.is_complete:
                    session.completed_at = datetime.now(timezone.utc).isoformat()

                return cp

        raise ValueError(f"Checkpoint {checkpoint_id} not found in session {session.session_id}")

    def get_audit_log(self, session: ReviewSession) -> dict:
        """Generate an audit log for the review session."""
        return {
            "session_id": session.session_id,
            "started_at": session.started_at,
            "completed_at": session.completed_at,
            "reviewer_id": session.reviewer_id,
            "total_checkpoints": session.total_checkpoints,
            "resolved": session.resolved_count,
            "pending": session.pending_count,
            "completion_percentage": session.completion_percentage,
            "checkpoints": [
                {
                    "id": cp.id,
                    "field": cp.field_name,
                    "priority": cp.priority.value,
                    "decision": cp.decision.value,
                    "confidence": cp.confidence,
                    "reviewed_at": cp.reviewed_at,
                    "had_revision": cp.revised_value is not None,
                }
                for cp in session.checkpoints
            ],
        }

    def _generate_checkpoints(self, soap_note, uncertainty_zones: list) -> list:
        """Auto-generate checkpoints from SOAP note content."""
        checkpoints = []

        # Always checkpoint Assessment (diagnosis)
        if hasattr(soap_note, 'assessment') and soap_note.assessment:
            self._checkpoint_counter += 1
            checkpoints.append(Checkpoint(
                id=f"cp_{self._checkpoint_counter:04d}",
                field_name="diagnosis",
                original_value=soap_note.assessment,
                confidence=0.6,
                priority=ReviewPriority.URGENT,
                reason="All AI-generated diagnoses require physician verification",
            ))

        # Always checkpoint Plan (medications)
        if hasattr(soap_note, 'plan') and soap_note.plan:
            self._checkpoint_counter += 1
            checkpoints.append(Checkpoint(
                id=f"cp_{self._checkpoint_counter:04d}",
                field_name="medication_plan",
                original_value=soap_note.plan,
                confidence=0.5,
                priority=ReviewPriority.URGENT,
                reason="Medication plans require physician approval — AI does not prescribe",
            ))

        # Checkpoint billing codes if present
        if hasattr(soap_note, 'billing_codes') and soap_note.billing_codes:
            self._checkpoint_counter += 1
            billing_str = json.dumps(soap_note.billing_codes, indent=2) if isinstance(soap_note.billing_codes, dict) else str(soap_note.billing_codes)
            checkpoints.append(Checkpoint(
                id=f"cp_{self._checkpoint_counter:04d}",
                field_name="billing_codes",
                original_value=billing_str,
                confidence=0.7,
                priority=ReviewPriority.STANDARD,
                reason="Billing codes are suggestions and require professional verification",
            ))

        # Add checkpoints from uncertainty zones
        for zone in uncertainty_zones:
            self._checkpoint_counter += 1
            field_name = zone.field if hasattr(zone, 'field') else "general"
            content = zone.content if hasattr(zone, 'content') else str(zone)
            confidence = zone.confidence if hasattr(zone, 'confidence') else 0.5

            # Determine priority based on field
            priority = ReviewPriority.STANDARD
            if any(kw in field_name.lower() for kw in self.critical_fields):
                priority = ReviewPriority.URGENT

            checkpoints.append(Checkpoint(
                id=f"cp_{self._checkpoint_counter:04d}",
                field_name=field_name,
                original_value=content,
                confidence=confidence,
                priority=priority,
                reason=f"Flagged as uncertain by extraction pipeline (confidence: {confidence:.2f})",
            ))

        return checkpoints
