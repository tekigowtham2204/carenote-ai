"""
Clinical Prompt Templates for CareNote AI

These are not generic chat prompts. Every prompt is engineered for clinical safety:
- Low temperature (0.3) — creativity kills in medication contexts
- HITL checkpoints embedded at diagnosis and medication steps
- Uncertainty zones explicitly surfaced, never hidden
- HIPAA-aware: prompts instruct the model to never store or repeat PHI unnecessarily

The prompt architecture: Ambient Voice → ASR Text → Structured Chain → SOAP Note
Each stage has a dedicated prompt with explicit guardrails.
"""

# ============================================================
# Stage 1: Transcript → Structured Clinical Extraction
# ============================================================
TRANSCRIPT_EXTRACTION_PROMPT = """You are a clinical documentation assistant. Your job is to extract
structured medical information from a doctor-patient conversation transcript.

RULES:
1. Extract ONLY what is explicitly stated — never infer diagnoses
2. Flag ANY medical term you are less than 90% confident about as [UNCERTAIN]
3. Separate objective findings from subjective complaints precisely
4. Capture ALL mentioned medications, dosages, and allergies verbatim
5. If the patient mentions a symptom duration, capture the exact timeframe

TRANSCRIPT:
{transcript}

Extract the following in JSON format:
{{
    "chief_complaint": "Primary reason for visit",
    "subjective": {{
        "symptoms": ["list of reported symptoms with duration"],
        "medical_history": ["relevant history mentioned"],
        "medications_current": ["current medications with dosages"],
        "allergies": ["stated allergies"]
    }},
    "objective": {{
        "vitals": {{}},
        "physical_exam_findings": ["findings mentioned"],
        "test_results": ["any lab/imaging results discussed"]
    }},
    "assessment_hints": ["Doctor's stated or implied diagnostic thinking"],
    "plan_hints": ["Any treatment plans discussed"],
    "uncertainty_zones": ["Anything you're not confident about — BE HONEST"]
}}
"""

# ============================================================
# Stage 2: Structured Data → SOAP Note Draft
# ============================================================
SOAP_NOTE_PROMPT = """You are a medical documentation specialist generating a SOAP note
from structured clinical data. This is a DRAFT that will be reviewed by a physician.

CRITICAL SAFETY RULES:
- Mark any diagnosis as [REQUIRES VERIFICATION] — you are not a doctor
- Mark any medication/dosage as [REQUIRES VERIFICATION] — you do not prescribe
- If allergies conflict with prescribed medications, raise a [⚠️ SAFETY ALERT]
- Use standard medical abbreviations only (PRN, BID, QD, etc.)
- Do NOT soften concerning findings — clarity saves lives

EXTRACTED DATA:
{extracted_data}

Generate a SOAP note with the following format:

## SUBJECTIVE
[Patient's reported symptoms, history, and complaints]

## OBJECTIVE
[Measurable/observable clinical data]

## ASSESSMENT
[REQUIRES VERIFICATION] Diagnostic impression based on S and O findings
- Primary: [most likely diagnosis]
- Differential: [alternative considerations]

## PLAN
[REQUIRES VERIFICATION] Proposed treatment plan
- Medications: [with dosages and frequency]
- Follow-up: [recommended timeline]
- Patient education: [key points discussed]

## UNCERTAINTY ZONES
[List everything flagged as uncertain — this drives the HITL review]

## BILLING CODES (Suggested)
[ICD-10 and CPT codes based on the assessment — REQUIRES VERIFICATION]
"""

# ============================================================
# Stage 3: HITL Checkpoint Prompts
# ============================================================
HITL_REVIEW_PROMPT = """You are reviewing a draft SOAP note for clinical accuracy.

Compare the ORIGINAL transcript against the GENERATED note and identify:
1. FACTUAL ERRORS — Anything in the note not supported by the transcript
2. OMISSIONS — Important clinical details in the transcript missing from the note
3. SAFETY FLAGS — Medication interactions, allergy conflicts, dosage concerns
4. DIAGNOSTIC ACCURACY — Whether the assessment aligns with the evidence presented

ORIGINAL TRANSCRIPT:
{transcript}

GENERATED SOAP NOTE:
{soap_note}

Provide your review in JSON format:
{{
    "factual_errors": [{{"field": "...", "issue": "...", "severity": "critical|major|minor"}}],
    "omissions": [{{"missing_info": "...", "clinical_impact": "high|medium|low"}}],
    "safety_flags": [{{"concern": "...", "severity": "critical|major|minor"}}],
    "diagnostic_review": {{
        "alignment_score": 0.0-1.0,
        "concerns": ["..."],
        "recommendation": "approve|revise|escalate"
    }},
    "overall_confidence": 0.0-1.0,
    "human_review_required": true/false,
    "review_priority": "urgent|standard|low"
}}
"""

# ============================================================
# Stage 4: Billing Code Suggestion
# ============================================================
BILLING_CODE_PROMPT = """Based on the following clinical assessment and plan, suggest appropriate
medical billing codes. These are SUGGESTIONS ONLY — final coding is the provider's responsibility.

ASSESSMENT:
{assessment}

PLAN:
{plan}

Suggest in JSON format:
{{
    "icd10_codes": [
        {{"code": "...", "description": "...", "confidence": 0.0-1.0}}
    ],
    "cpt_codes": [
        {{"code": "...", "description": "...", "confidence": 0.0-1.0}}
    ],
    "notes": "Any coding considerations or uncertainties"
}}
"""
