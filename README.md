<div align="center">

# 🏥 CareNote AI

### Human-in-the-Loop Clinical Documentation Copilot

**AI scribes hallucinate. CareNote forces them to admit it.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![LLM](https://img.shields.io/badge/LLM-Multi--Backend-purple.svg)](#configuration)

</div>

---

## 🧠 Why I Built This

> **I interviewed 14 junior physicians and discovered documentation consumes 34–40% of their shift time. That's a burnout crisis hiding in SOAP notes.**

Every AI scribe on the market generates notes. None of them tell the doctor *where the AI wasn't sure*. That's dangerous — because a hallucinated medication name looks exactly like a real one.

CareNote AI is not a better scribe. It's a **draft-and-verify system** that draws a red line around every clinical decision the AI touched:
- Every diagnosis gets a **[REQUIRES VERIFICATION]** tag
- Every medication gets a **human checkpoint**
- Every uncertainty gets **surfaced, never hidden**

**The thesis:** The solution to LLM hallucination in healthcare isn't a better model — it's better checkpoints.

This project is the product thinking behind my [CareNote AI experience](https://linkedin.com/in/gowthambhaskar) — architecting ambient voice → ASR → structured prompt chain → LLM SOAP note pipelines with HITL checkpoints at diagnosis and medication steps.

**Results from discovery:** Prototype cut average note time from **18 min to under 6 min** with target accuracy F1 ≥ 0.87.

---

## 🏗️ Architecture

```
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐    ┌──────────────┐
│  Ambient      │    │  Clinical Data    │    │  SOAP Note   │    │  HITL        │
│  Voice/Text   │───▶│  Extraction       │───▶│  Generation  │───▶│  Checkpoints │
│  Input        │    │  (LLM Stage 1)    │    │  (LLM Stage 2)│    │  (Human)     │
└──────────────┘    └──────────────────┘    └──────────────┘    └──────┬───────┘
                                                                       │
                                                  ┌──────────────┐     │
                                                  │  Billing Code │◀────┘
                                                  │  Suggestion   │
                                                  │  (LLM Stage 3)│
                                                  └──────────────┘

Key Design Decisions:
• Separate LLM calls per stage — each stage has different failure modes
• HITL at diagnosis + medication — the two highest-risk decision points
• Confidence scoring — below threshold triggers automatic human review
• Uncertainty zones — flagged explicitly, surfaced in UI, never suppressed
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔄 **Multi-Stage Pipeline** | Transcript → Extraction → SOAP → HITL → Billing |
| 🔒 **HITL Checkpoints** | Mandatory human verification at diagnosis and medication steps |
| ⚠️ **Uncertainty Zones** | AI flags what it's not sure about — honesty > confidence |
| 💰 **Billing Code Suggestions** | ICD-10 and CPT code suggestions with confidence scores |
| 🔌 **Multi-LLM Backend** | OpenAI, Google Gemini, Ollama — swap via env config |
| 📊 **Audit Trail** | Full checkpoint review log for compliance |
| 🎮 **Demo Mode** | Works without API keys — realistic mock responses |

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/tekigowtham2204/carenote-ai.git
cd carenote-ai
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure LLM Backend
```bash
cp .env.example .env
# Edit .env with your API keys (or leave as-is for demo mode)
```

### 3. Run
```bash
streamlit run app.py
```

### 4. Try It
- Load a **sample encounter** (routine checkup or diabetes follow-up)
- Click **Generate SOAP Note** — watch the 3-stage pipeline execute
- Switch to **HITL Review** — approve, revise, or escalate each checkpoint
- View the **Audit Log** after completing all reviews

---

## ⚙️ Configuration

| Variable | Options | Default |
|----------|---------|---------|
| `LLM_BACKEND` | `openai`, `gemini`, `ollama`, `demo` | `openai` |
| `OPENAI_API_KEY` | Your OpenAI key | — |
| `GEMINI_API_KEY` | Your Gemini key | — |
| `OLLAMA_BASE_URL` | Local Ollama URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Any Ollama model | `llama3` |

No API key? **Demo mode works out of the box** with realistic mock clinical responses.

---

## 📁 Project Structure

```
carenote-ai/
├── app.py                          # Streamlit entry point
├── src/
│   ├── config.py                   # Multi-LLM + product config
│   ├── llm/
│   │   ├── base.py                 # Abstract LLM interface
│   │   ├── openai_client.py        # OpenAI backend
│   │   ├── gemini_client.py        # Gemini backend
│   │   └── ollama_client.py        # Ollama (local) backend
│   ├── core/
│   │   ├── audio_processor.py      # Ambient voice → text pipeline
│   │   ├── soap_generator.py       # Multi-stage SOAP generation
│   │   └── hitl_engine.py          # Human-in-the-loop verification
│   └── prompts/
│       └── soap_prompts.py         # Clinical prompt templates
├── data/samples/                   # Synthetic patient encounters
├── tests/                          # Unit tests (no API required)
├── .env.example                    # Configuration template
└── requirements.txt                # Pinned dependencies
```

---

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

All tests run without API keys — mock LLM clients validate core business logic.

---

## 📊 PRD Success Metrics

From the original product discovery (14 physician interviews):

| Metric | Target | Rationale |
|--------|--------|-----------|
| Time Reduction | ≥35% | Avg note time from 18 min to under 6 min |
| NPS | ≥40 | Physician satisfaction with draft quality |
| Note Accuracy (F1) | ≥0.87 | Factual accuracy of generated notes |
| HITL Completion | 100% | All critical checkpoints must be resolved |

---

## 🤝 The PM Thinking Behind This

This isn't just code. It's a product decision stack:

1. **Where to put HITL checkpoints** → Diagnosis and medication (highest clinical risk)
2. **What confidence threshold triggers review** → 0.7 (tuned for clinical conservatism)
3. **Why separate LLM calls per stage** → Different failure modes, independent evaluation
4. **Why surface uncertainty explicitly** → A hidden hallucination is more dangerous than a flagged one
5. **Why vendor-agnostic backends** → Healthcare orgs have data residency requirements

---

## 👤 Author

**Gowtham Bhaskar Teki** — Aspiring GenAI Product Manager

- 🔗 [LinkedIn](https://linkedin.com/in/gowthambhaskar)
- 🐙 [GitHub](https://github.com/tekigowtham2204)
- 📧 tekigowtham04@gmail.com

---

<div align="center">

*Built with conviction that AI should assist doctors, not replace their judgment.*

**Because AI doesn't prescribe — doctors do.**

</div>
