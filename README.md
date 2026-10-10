# 🏥 AI-Driven Public Health Chatbot for Disease Awareness

A multi-agent AI chatbot that provides reliable, safe, and easy-to-understand
health information by combining a curated knowledge base (RAG) with live
web search, wrapped in safety guardrails for emergency detection.

---

## Problem Statement

Build an AI-driven chatbot that raises public awareness about diseases —
their symptoms, prevention, and treatment — using natural language
interaction, while safely directing users to professional medical help
when needed.

---

## How it works (Architecture)

```
                     User Question
                          │
                          ▼
                 ┌─────────────────┐
                 │  Safety Agent   │  → is it health-related? emergency? safe?
                 └────────┬────────┘
                          │ (if health-related & safe)
              ┌───────────┴────────────┐
              ▼                        ▼
      ┌───────────────┐       ┌────────────────┐
      │   RAG Agent    │       │  Search Agent  │
      │ (knowledge     │       │ (Tavily web     │
      │  base lookup)  │       │  search)        │
      └───────┬────────┘       └────────┬────────┘
              │                          │
              └────────────┬─────────────┘
                            ▼
                  ┌────────────────────┐
                  │  Responder Agent   │  → combines context, writes
                  │  (LLM via Groq)    │     final empathetic answer
                  └────────┬───────────┘
                            │
                            ▼
                     Final Response
              (with emergency banner if needed)
```

---

## ML symptom-pattern layer

Between the Safety Agent and the RAG/Search agents, an ML step helps with
overlapping symptoms:

1. The LLM maps the user's text to a fixed symptom vocabulary (JSON list).
2. A calibrated Random Forest returns the top-3 conditions with probabilities.
3. Decision logic: fewer than 2 symptoms -> normal answer; clear winner (3+
   symptoms, high probability and margin) -> "consistent with X"; overlap ->
   the bot asks ONE targeted follow-up (e.g. "Do you also have chills?") and
   remembers the answer across turns; still unclear -> "see a doctor".
4. The result is passed to the responder as advisory context, never as a diagnosis.
5. A deterministic red-flag keyword check runs alongside the LLM emergency check.

```bash
python -m ml.train                      # starter model (synthetic, from the KB)
python -m ml.train --csv Training.csv   # real dataset (Kaggle format)
python test_ml.py                       # offline tests, no API keys
```

The starter model is generated from the symptom profiles in `ml/symptoms.py`,
so its accuracy figures are not real-world performance. Train on a real
dataset before relying on it.

---

## Tech Stack

| Component       | Technology                              |
|------------------|------------------------------------------|
| LLM              | Groq API (Qwen 3 32B)                    |
| RAG              | LangChain + ChromaDB + Sentence Transformers |
| Web Search       | Tavily Search API                        |
| UI               | Streamlit                                |
| Deployment       | Ngrok (for Colab) / Streamlit Cloud / Render |

---

## Project Structure

```
HealthChatbot/
├── app.py                  ← Streamlit UI (main entry point)
├── pipeline.py             ← Orchestrates all agents
├── test_pipeline.py        ← CLI script for quick testing
├── colab_launcher.py       ← Instructions for running in Google Colab
├── requirements.txt        ← Python dependencies
├── agents/
│   ├── rag_agent.py        ← Searches static health knowledge base
│   ├── search_agent.py     ← Searches the web for latest info
│   ├── safety.py           ← Safety / emergency / scope classifier
│   └── responder.py        ← Generates the final combined answer
└── data/
    └── health_kb.txt       ← Curated health knowledge base (9 diseases + tips)
```

---

## Setup & Run Instructions

### Option A — Run locally

```bash
cd HealthChatbot
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal, and enter your API keys
in the sidebar.

### Option B — Run in Google Colab

1. Upload the whole `HealthChatbot` folder to Colab (or zip and unzip it).
2. Open `colab_launcher.py` for cell-by-cell instructions.
3. Run the install cell, then the launch cell with your ngrok token.
4. Open the public URL it prints.

---

## API Keys Needed (all free tier)

- **Groq API Key** → https://console.groq.com
- **Tavily API Key** → https://tavily.com
- **Ngrok Auth Token** (only for Colab deployment) → https://dashboard.ngrok.com

---

## Features

- ✅ Multi-agent pipeline (Safety → RAG → Web Search → Responder)
- ✅ Covers 9 major diseases: Dengue, Malaria, Typhoid, COVID-19, TB,
  Cholera, Diabetes, Hypertension + general health tips
- ✅ Emergency detection with automatic warning banner and helpline numbers
- ✅ Out-of-scope question redirection (stays focused on health topics)
- ✅ Live web search for recent health news/outbreaks
- ✅ Clean chat interface with quick-topic buttons
- ✅ Medical disclaimer always shown

---

## Sample Interaction

```
User: What are the symptoms of dengue fever?

Bot: Dengue fever typically causes:
     • High fever (around 104°F)
     • Severe headache and pain behind the eyes
     • Joint and muscle pain
     • Skin rash
     • Mild bleeding from nose/gums

     Seek medical care immediately if you experience severe
     abdominal pain, persistent vomiting, or difficulty breathing.

     Remember: Always consult a qualified doctor for personal
     medical advice.
```

```
User: I am having severe chest pain and difficulty breathing!

Bot: 🚨 THIS MAY BE A MEDICAL EMERGENCY 🚨
     📞 Call 108 (Ambulance) immediately
     🏥 Go to the nearest hospital / emergency room now
     Do not wait — act immediately.

     [followed by relevant safety information]
```

---

## Future Improvements

- Add multilingual support (Hindi, Telugu, etc.) for wider accessibility
- Integrate a symptom-checker decision tree for structured triage
- Add a nearby-hospital locator using Google Maps API
- Voice input/output for accessibility
- Expand knowledge base with more diseases and government health schemes

---

## Disclaimer

This chatbot provides general health information for educational purposes
only. It is **not** a substitute for professional medical advice, diagnosis,
or treatment. Always consult a qualified healthcare provider for any medical
concerns.
