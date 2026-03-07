"""
CareNote AI — Streamlit Application

The main entry point. This is where the clinical workflow comes to life.

Architecture: Ambient Voice → ASR → Structured Prompt Chain → LLM SOAP Note → HITL Verification
"""

import streamlit as st
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import get_config, LLMConfig
from src.core.audio_processor import AudioProcessor
from src.core.soap_generator import SOAPGenerator
from src.core.hitl_engine import HITLEngine, ReviewDecision, ReviewPriority
from src.llm.base import BaseLLMClient, LLMResponse


# ============================================================
# Demo LLM Client (works without API keys)
# ============================================================
class DemoLLMClient(BaseLLMClient):
    """Demo client that generates realistic mock responses.
    
    This allows the full pipeline to work without API keys.
    In production, swap this for OpenAI/Gemini/Ollama via config.
    """

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.3) -> LLMResponse:
        # Detect which stage of the pipeline we're in
        if "Extract the following in JSON format" in prompt:
            return self._mock_extraction(prompt)
        elif "Generate a SOAP note" in prompt:
            return self._mock_soap_note(prompt)
        elif "suggest appropriate medical billing codes" in prompt.lower():
            return self._mock_billing(prompt)
        elif "reviewing a draft SOAP note" in prompt.lower():
            return self._mock_review(prompt)
        else:
            return LLMResponse(content="[Demo mode] — Configure an LLM backend for real responses.", model="demo")

    def is_available(self) -> bool:
        return True

    def _mock_extraction(self, prompt: str) -> LLMResponse:
        data = {
            "chief_complaint": "Persistent headaches for approximately two weeks, primarily frontal, typically afternoon onset",
            "subjective": {
                "symptoms": ["Frontal headaches x 2 weeks, afternoon predominance", "Pain severity 6-7/10", "Photosensitivity present", "Sleep deprivation (4-5 hrs/night)"],
                "medical_history": ["Family history: maternal migraines", "No personal migraine diagnosis"],
                "medications_current": ["Multivitamins (daily)", "Ibuprofen PRN for headaches"],
                "allergies": ["Sulfa drugs — manifested as rash (historical)"]
            },
            "objective": {
                "vitals": {"BP": "128/82 mmHg", "HR": "76 bpm", "Temp": "98.4°F"},
                "physical_exam_findings": ["Neurological exam normal", "Cranial nerves intact", "No papilledema"],
                "test_results": []
            },
            "assessment_hints": ["Tension-type headache pattern", "Sleep deprivation as likely exacerbating factor"],
            "plan_hints": ["Sleep hygiene counseling", "Continue ibuprofen 400mg TID max", "Consider amitriptyline if persistent", "3-week follow-up"],
            "uncertainty_zones": ["Migraine vs tension-type differentiation — family history warrants monitoring"]
        }
        return LLMResponse(content=json.dumps(data, indent=2), model="demo", confidence=0.85)

    def _mock_soap_note(self, prompt: str) -> LLMResponse:
        note = """## SUBJECTIVE
Patient presents with a 2-week history of persistent frontal headaches, typically onset in the afternoon, rated 6-7/10 severity. Reports associated photosensitivity. Denies nausea. Significant sleep deprivation noted (4-5 hours/night) attributed to work-related stress. Family history positive for maternal migraines; no personal history of diagnosed migraines. Current medications include daily multivitamins and PRN ibuprofen. Known allergy to sulfa drugs (rash reaction).

## OBJECTIVE
Vitals: BP 128/82 mmHg, HR 76 bpm, Temp 98.4°F
Neurological examination within normal limits. Cranial nerves II-XII intact. No papilledema on fundoscopic examination.

## ASSESSMENT
[REQUIRES VERIFICATION] Primary: Tension-type headache (G44.209) — episodic pattern with identifiable trigger (sleep deprivation)
Differential: Migraine without aura — warranted by family history and photosensitivity features
Note: Sleep deprivation is a likely primary contributor and should be addressed before pharmacological escalation.

## PLAN
[REQUIRES VERIFICATION]
1. Sleep hygiene counseling — target 7-8 hours/night; discussed sleep environment optimization
2. Continue Ibuprofen 400mg PO q8h PRN (max 1200mg/day) — ⚠️ Note: sulfa allergy documented, no NSAID cross-reactivity concern
3. If headaches persist beyond 3 weeks: initiate Amitriptyline 10mg PO QHS
4. Follow-up in 3 weeks for symptom reassessment
5. Return precautions: visual changes, thunderclap headache, fever, neck stiffness

## UNCERTAINTY ZONES
- Migraine vs. tension-type differentiation requires further monitoring given family history
- Sleep deprivation as sole trigger vs. evolving migraine pattern to be reassessed at follow-up

## BILLING CODES (Suggested)
[REQUIRES VERIFICATION]
- ICD-10: G44.209 (Tension-type headache, unspecified, not intractable)
- CPT: 99214 (Office visit, established patient, moderate complexity)"""

        return LLMResponse(content=note, model="demo", confidence=0.82)

    def _mock_billing(self, prompt: str) -> LLMResponse:
        codes = {
            "icd10_codes": [
                {"code": "G44.209", "description": "Tension-type headache, unspecified, not intractable", "confidence": 0.85},
                {"code": "G47.00", "description": "Insomnia, unspecified (secondary)", "confidence": 0.70}
            ],
            "cpt_codes": [
                {"code": "99214", "description": "Office visit, established patient, moderate complexity", "confidence": 0.90}
            ],
            "notes": "Primary code based on documented tension-type impression. Consider adding insomnia code given documented 4-5hr sleep pattern."
        }
        return LLMResponse(content=json.dumps(codes, indent=2), model="demo", confidence=0.85)

    def _mock_review(self, prompt: str) -> LLMResponse:
        review = {
            "factual_errors": [],
            "omissions": [
                {"missing_info": "Patient's occupation/work type not documented", "clinical_impact": "low"}
            ],
            "safety_flags": [
                {"concern": "Ibuprofen dosage ceiling correctly documented. Sulfa allergy noted with no cross-reactivity concern — accurate.", "severity": "minor"}
            ],
            "diagnostic_review": {
                "alignment_score": 0.88,
                "concerns": ["Family migraine history warrants lower threshold for migraine diagnosis at follow-up"],
                "recommendation": "approve"
            },
            "overall_confidence": 0.88,
            "human_review_required": True,
            "review_priority": "standard"
        }
        return LLMResponse(content=json.dumps(review, indent=2), model="demo", confidence=0.88)


# ============================================================
# App Configuration
# ============================================================
def get_llm_client(config):
    """Get the appropriate LLM client based on configuration."""
    if config.llm.backend == "openai" and config.llm.openai_api_key:
        from src.llm.openai_client import OpenAIClient
        return OpenAIClient(api_key=config.llm.openai_api_key, model=config.llm.openai_model)
    elif config.llm.backend == "gemini" and config.llm.gemini_api_key:
        from src.llm.gemini_client import GeminiClient
        return GeminiClient(api_key=config.llm.gemini_api_key, model=config.llm.gemini_model)
    elif config.llm.backend == "ollama":
        from src.llm.ollama_client import OllamaClient
        return OllamaClient(base_url=config.llm.ollama_base_url, model=config.llm.ollama_model)
    else:
        return DemoLLMClient()


# ============================================================
# Streamlit UI
# ============================================================
def main():
    st.set_page_config(
        page_title="CareNote AI",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            color: #64748b;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        .checkpoint-urgent {
            border-left: 4px solid #ef4444;
            padding: 1rem;
            background: #fef2f2;
            border-radius: 0 8px 8px 0;
            margin: 0.5rem 0;
        }
        .checkpoint-standard {
            border-left: 4px solid #f59e0b;
            padding: 1rem;
            background: #fffbeb;
            border-radius: 0 8px 8px 0;
            margin: 0.5rem 0;
        }
        .checkpoint-low {
            border-left: 4px solid #10b981;
            padding: 1rem;
            background: #f0fdf4;
            border-radius: 0 8px 8px 0;
            margin: 0.5rem 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .soap-section {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">🏥 CareNote AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Human-in-the-Loop Clinical Documentation Copilot — Draft, Verify, Approve</p>', unsafe_allow_html=True)

    # Initialize
    config = get_config()
    llm_client = get_llm_client(config)
    audio_processor = AudioProcessor(samples_dir="data/samples")
    soap_generator = SOAPGenerator(llm_client=llm_client)
    hitl_engine = HITLEngine(
        confidence_threshold=config.uncertainty_threshold,
        critical_fields=config.critical_fields,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        backend = st.selectbox(
            "LLM Backend",
            ["demo", "openai", "gemini", "ollama"],
            help="Select your LLM backend. Demo mode works without API keys.",
        )

        if backend != "demo":
            st.info(f"Configure `{backend.upper()}` credentials in `.env`")

        st.divider()
        st.markdown("### 📊 PRD Success Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Target Time ↓", "≥35%")
            st.metric("NPS Target", "≥40")
        with col2:
            st.metric("F1 Accuracy", "≥0.87")
            st.metric("HITL Threshold", f"{config.uncertainty_threshold}")

        st.divider()
        st.markdown("### 🔬 About")
        st.markdown("""
        **Built by [Gowtham Bhaskar Teki](https://github.com/tekigowtham2204)**

        CareNote AI turns AI scribes into draft-and-verify tools.
        Every diagnosis and medication goes through a human checkpoint.

        _Because AI doesn't prescribe — doctors do._
        """)

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📝 Generate Note", "✅ HITL Review", "📈 Pipeline Metrics"])

    # =========================================================
    # Tab 1: Generate Note
    # =========================================================
    with tab1:
        st.markdown("### Input: Doctor-Patient Encounter")

        input_method = st.radio(
            "Input Method",
            ["Type/Paste Transcript", "Load Sample Encounter"],
            horizontal=True,
        )

        transcript = ""

        if input_method == "Load Sample Encounter":
            samples_file = "data/samples/encounters.json"
            if os.path.exists(samples_file):
                with open(samples_file) as f:
                    samples = json.load(f)
                sample_names = [s["sample_name"] for s in samples]
                selected = st.selectbox("Select Encounter", sample_names)
                selected_sample = next(s for s in samples if s["sample_name"] == selected)
                transcript = selected_sample["full_text"]
                st.text_area("Transcript Preview", transcript, height=250, disabled=True)
            else:
                st.warning("Sample data not found. Use text input instead.")
                input_method = "Type/Paste Transcript"

        if input_method == "Type/Paste Transcript":
            transcript = st.text_area(
                "Paste Doctor-Patient Conversation",
                placeholder="Doctor: What brings you in today?\nPatient: I've been having headaches...",
                height=250,
            )

        if st.button("🚀 Generate SOAP Note", type="primary", disabled=not transcript):
            with st.spinner("Processing clinical encounter..."):
                # Run the pipeline
                processed = audio_processor.process_text_input(transcript)

                # Stage 1: Extract
                with st.status("Stage 1: Extracting clinical data...", expanded=True) as status:
                    extracted = soap_generator.extract_clinical_data(processed.full_text)
                    status.update(label="✅ Clinical data extracted", state="complete")

                # Stage 2: Generate SOAP
                with st.status("Stage 2: Generating SOAP note...", expanded=True) as status:
                    soap_note = soap_generator.generate_soap_note(extracted)
                    status.update(label="✅ SOAP note generated", state="complete")

                # Stage 3: Billing codes
                with st.status("Stage 3: Suggesting billing codes...", expanded=True) as status:
                    if soap_note.assessment:
                        soap_note.billing_codes = soap_generator.suggest_billing_codes(
                            soap_note.assessment, soap_note.plan
                        )
                    status.update(label="✅ Billing codes suggested", state="complete")

                # Store in session
                st.session_state["soap_note"] = soap_note
                st.session_state["extracted_data"] = extracted
                st.session_state["transcript"] = transcript

                # Create HITL review session
                review_session = hitl_engine.create_review_session(
                    soap_note, soap_note.uncertainty_zones
                )
                st.session_state["review_session"] = review_session

            # Display results
            st.success("✅ SOAP note generated! Review the output below, then proceed to **HITL Review** tab.")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📋 SOAP Note Draft")
                st.markdown(f'<div class="soap-section"><strong>SUBJECTIVE</strong><br>{soap_note.subjective}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="soap-section"><strong>OBJECTIVE</strong><br>{soap_note.objective}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="soap-section"><strong>ASSESSMENT</strong><br>{soap_note.assessment}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="soap-section"><strong>PLAN</strong><br>{soap_note.plan}</div>', unsafe_allow_html=True)

            with col2:
                st.markdown("#### ⚠️ Uncertainty Zones")
                if soap_note.uncertainty_zones:
                    for zone in soap_note.uncertainty_zones:
                        content = zone.content if hasattr(zone, 'content') else str(zone)
                        st.warning(content)
                else:
                    st.info("No uncertainty zones flagged.")

                st.markdown("#### 💰 Billing Codes (Suggested)")
                if soap_note.billing_codes and not soap_note.billing_codes.get("parse_error"):
                    st.json(soap_note.billing_codes)
                else:
                    st.info("Billing codes pending.")

    # =========================================================
    # Tab 2: HITL Review
    # =========================================================
    with tab2:
        st.markdown("### ✅ Human-in-the-Loop Review")

        if "review_session" not in st.session_state:
            st.info("Generate a SOAP note first, then review checkpoints here.")
        else:
            session = st.session_state["review_session"]

            # Progress
            progress = session.completion_percentage / 100
            st.progress(progress, text=f"Review Progress: {session.resolved_count}/{session.total_checkpoints} checkpoints resolved")

            critical = session.critical_pending
            if critical:
                st.error(f"⚠️ {len(critical)} URGENT checkpoint(s) require immediate review")

            # Display checkpoints
            for i, cp in enumerate(session.checkpoints):
                priority_class = f"checkpoint-{cp.priority.value}"
                priority_emoji = {"urgent": "🔴", "standard": "🟡", "low": "🟢"}[cp.priority.value]

                with st.expander(f"{priority_emoji} [{cp.priority.value.upper()}] {cp.field_name} — {cp.id}", expanded=not cp.is_resolved):
                    st.markdown(f"**Reason:** {cp.reason}")
                    st.markdown(f"**AI Confidence:** {cp.confidence:.0%}")
                    st.text_area(f"Content ({cp.id})", cp.original_value, height=150, disabled=True, key=f"content_{cp.id}")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"✅ Approve", key=f"approve_{cp.id}"):
                            hitl_engine.resolve_checkpoint(session, cp.id, ReviewDecision.APPROVED)
                            st.rerun()
                    with col2:
                        if st.button(f"✏️ Revise", key=f"revise_{cp.id}"):
                            hitl_engine.resolve_checkpoint(session, cp.id, ReviewDecision.REVISED, reviewer_notes="Physician revision required")
                            st.rerun()
                    with col3:
                        if st.button(f"🚨 Escalate", key=f"escalate_{cp.id}"):
                            hitl_engine.resolve_checkpoint(session, cp.id, ReviewDecision.ESCALATED, reviewer_notes="Escalated for senior review")
                            st.rerun()

                    if cp.is_resolved:
                        st.success(f"Decision: {cp.decision.value.upper()} at {cp.reviewed_at}")

            if session.is_complete:
                st.balloons()
                st.success("🎉 All checkpoints reviewed! Note is ready for finalization.")

                # Show audit log
                with st.expander("📋 Audit Log"):
                    audit = hitl_engine.get_audit_log(session)
                    st.json(audit)

    # =========================================================
    # Tab 3: Pipeline Metrics
    # =========================================================
    with tab3:
        st.markdown("### 📈 Pipeline Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Notes Generated", st.session_state.get("notes_generated", 1 if "soap_note" in st.session_state else 0))
        with col2:
            st.metric("Avg Generation Time", "< 6 min" if "soap_note" in st.session_state else "—")
        with col3:
            if "review_session" in st.session_state:
                st.metric("HITL Completion", f"{st.session_state['review_session'].completion_percentage:.0f}%")
            else:
                st.metric("HITL Completion", "—")
        with col4:
            st.metric("Backend", backend.upper())

        st.divider()
        st.markdown("### 🏗️ Pipeline Architecture")
        st.markdown("""
        ```
        ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐    ┌──────────────┐
        │  Ambient      │    │  Clinical Data    │    │  SOAP Note   │    │  HITL        │
        │  Voice/Text   │───▶│  Extraction       │───▶│  Generation  │───▶│  Checkpoints │
        │  Input        │    │  (LLM Stage 1)    │    │  (LLM Stage 2)│    │  (Human)     │
        └──────────────┘    └──────────────────┘    └──────────────┘    └──────────────┘
                                                                                │
                                                         ┌──────────────┐       │
                                                         │  Billing Code │◀──────┘
                                                         │  Suggestion   │
                                                         │  (LLM Stage 3)│
                                                         └──────────────┘
        ```
        """)

        st.markdown("""
        **Key Design Decisions:**
        - 🔒 **Separate LLM calls per stage** — each stage has different failure modes
        - 🎯 **HITL at diagnosis + medication** — the two highest-risk decision points
        - 📊 **Confidence scoring** — below threshold triggers automatic human review
        - 🔄 **Vendor-agnostic** — swap OpenAI/Gemini/Ollama via environment config
        """)


if __name__ == "__main__":
    main()
