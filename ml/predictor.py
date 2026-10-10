"""
Symptom predictor: LLM symptom extraction + ML ranking + overlap handling.

Flow for one user message (see pipeline.py):
  1. extract_symptoms()  - LLM maps free text to the model's symptom vocabulary
  2. SymptomPredictor.assess() - ML top-3 + decision:
        need_more  : fewer than MIN_SYMPTOMS known -> ask for more details
        confident  : clear winner -> pass to the responder as "consistent with"
        ambiguous  : top candidates overlap -> ask ONE targeted follow-up
        uncertain  : still unclear after follow-ups -> general guidance + see a doctor
The ML output is advisory only; the responder must never present it as a diagnosis.
"""

import json
import os
import re
from dataclasses import dataclass, field

import joblib
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from ml.symptoms import SYMPTOM_LABELS
from ml.train import MODEL_PATH, make_synthetic, train

MIN_SYMPTOMS = 2          # below this we ask for more details instead of predicting
MIN_CONFIDENT_SYMPTOMS = 3  # fewer than this can never be "confident"
CONFIDENT_PROB = 0.55     # top-1 probability needed to call it confident
CONFIDENT_MARGIN = 0.20   # top-1 minus top-2 gap needed to call it confident
MAX_FOLLOWUPS = 3         # stop asking after this many follow-up questions

_YES = {"yes", "y", "yeah", "yep", "yup", "i do", "i have", "ha", "haan", "ho"}
_NO = {"no", "n", "nope", "nah", "i don't", "i do not", "nahi", "illa"}


@dataclass
class SymptomState:
    """Per-conversation memory; keep one in st.session_state."""
    symptoms: set = field(default_factory=set)
    asked: set = field(default_factory=set)
    pending: str = None  # symptom we just asked about

    def reset(self):
        self.symptoms.clear()
        self.asked.clear()
        self.pending = None


@dataclass
class Assessment:
    status: str                      # need_more | confident | ambiguous | uncertain
    top3: list                       # [(disease, prob), ...]
    symptoms: list
    followup_symptom: str = None
    followup_question: str = None


def _parse_json_list(text: str):
    text = text.split("</think>")[-1]
    m = re.search(r"\[.*?\]", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return [str(x) for x in data]
    except json.JSONDecodeError:
        return []


def extract_symptoms(question: str, llm, vocabulary: list) -> set:
    """Map free text to symptoms in `vocabulary` using the LLM (JSON list out)."""
    vocab = ", ".join(vocabulary)
    prompt = (
        "Extract the symptoms the USER says they currently have from the message.\n"
        f"Allowed symptom ids: {vocab}\n"
        "Rules: use only ids from the list; ignore symptoms the user says they do NOT "
        "have; ignore questions about other people's or general disease info unless the "
        "user describes their own symptoms. Reply with ONLY a JSON list, e.g. "
        '["high_fever","headache"] or [] if none.'
    )
    try:
        resp = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=question)])
        found = _parse_json_list(resp.content)
    except Exception:
        return set()
    allowed = set(vocabulary)
    return {s for s in found if s in allowed}


class SymptomPredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            X, y, syms = make_synthetic()
            train(X, y, syms, save_path=model_path, verbose=False)
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.symptoms = bundle["symptoms"]
        self.classes = list(bundle["classes"])
        self.freq = bundle["freq"]  # (n_classes, n_symptoms)
        self._idx = {s: i for i, s in enumerate(self.symptoms)}

    # ── prediction ──
    def predict_top3(self, symptoms) -> list:
        x = np.zeros((1, len(self.symptoms)))
        for s in symptoms:
            if s in self._idx:
                x[0, self._idx[s]] = 1
        p = self.model.predict_proba(x)[0]
        order = p.argsort()[::-1][:3]
        return [(self.classes[i], float(p[i])) for i in order]

    def _best_followup(self, top3, known, asked):
        """Symptom that best separates the top two candidates and isn't known yet."""
        a, b = (self.classes.index(top3[0][0]), self.classes.index(top3[1][0]))
        diff = np.abs(self.freq[a] - self.freq[b])
        for i in np.argsort(diff)[::-1]:
            s = self.symptoms[i]
            if s not in known and s not in asked and diff[i] > 0.15:
                return s
        return None

    def assess(self, state: SymptomState) -> Assessment:
        syms = sorted(state.symptoms)
        if len(syms) < MIN_SYMPTOMS:
            return Assessment("need_more", [], syms)

        top3 = self.predict_top3(syms)
        (_, p1), (_, p2) = top3[0], top3[1]
        if (len(syms) >= MIN_CONFIDENT_SYMPTOMS and p1 >= CONFIDENT_PROB
                and (p1 - p2) >= CONFIDENT_MARGIN):
            return Assessment("confident", top3, syms)

        if len(state.asked) < MAX_FOLLOWUPS:
            f = self._best_followup(top3, state.symptoms, state.asked)
            if f:
                label = SYMPTOM_LABELS.get(f, f.replace("_", " "))
                return Assessment("ambiguous", top3, syms, f,
                                  f"Do you also have {label}?")
        return Assessment("uncertain", top3, syms)


def update_state_from_message(question: str, state: SymptomState, llm,
                              predictor: SymptomPredictor):
    """
    Update `state` from a user message. A bare yes/no answers the pending
    follow-up question deterministically; anything else goes to the LLM extractor.
    """
    answer = question.strip().lower().rstrip(".!")
    if state.pending and answer in _YES | _NO:
        if answer in _YES:
            state.symptoms.add(state.pending)
        state.asked.add(state.pending)
        state.pending = None
        return
    state.pending = None
    state.symptoms |= extract_symptoms(question, llm, predictor.symptoms)


DISPLAY_NAMES = {
    "Peptic ulcer diseae": "Peptic ulcer disease",
    "Osteoarthristis": "Osteoarthritis",
    "Dimorphic hemmorhoids(piles)": "Piles (hemorrhoids)",
    "(vertigo) Paroymsal Positional Vertigo": "Vertigo (BPPV)",
    "Paralysis (brain hemorrhage)": "Brain hemorrhage with paralysis",
    "GERD": "GERD (acid reflux)",
}
SHOW_MIN_PROB = 0.10  # candidates below this are not shown to the user


def pretty(name: str) -> str:
    """Clean up dataset spellings for display."""
    name = " ".join(str(name).split())
    if name in DISPLAY_NAMES:
        return DISPLAY_NAMES[name]
    return name[:1].upper() + name[1:]


def _shown(a: Assessment):
    keep = [(d, p) for d, p in a.top3 if p >= SHOW_MIN_PROB]
    return keep or a.top3[:1]


def format_for_llm(a: Assessment) -> str:
    """Text block handed to the responder as grounded ML context."""
    if a.status == "need_more" or not a.top3:
        return ""
    names = [pretty(d) for d, _ in _shown(a)]
    syms = ", ".join(s.replace("_", " ").strip() for s in a.symptoms)
    base = (f"Symptoms reported by user: {syms}.\n"
            f"Closest matches from the symptom-pattern model, best first: "
            f"{', '.join(names)}.\n")
    if a.status == "confident":
        return base + (f"The symptoms most closely match {names[0]}. Say this directly "
                       "('your symptoms most closely match ...'), but never say the user "
                       "definitely has it, and recommend a doctor visit or test to confirm.")
    return base + ("The match is not certain because several conditions overlap. Name these "
                   "closest matches in order, say clearly that more information or a "
                   "doctor's exam is needed to tell them apart, and do not claim one "
                   "definitely.")


def format_matches(a: Assessment) -> str:
    """User-visible block listing the closest matches (added by code, not the LLM)."""
    if a is None or a.status == "need_more" or not a.top3:
        return ""
    lines = []
    for i, (d, _) in enumerate(_shown(a), 1):
        if i == 1:
            label = "strong match" if a.status == "confident" else "closest match"
        else:
            label = "possible match"
        lines.append(f"{i}. **{pretty(d)}** — {label}")
    syms = ", ".join(s.replace("_", " ").strip() for s in a.symptoms)
    return ("\n\n---\n**Closest matches for your symptoms** "
            "(from a symptom-pattern model, not a diagnosis):\n"
            + "\n".join(lines)
            + f"\n\n*Symptoms I understood: {syms}*")
