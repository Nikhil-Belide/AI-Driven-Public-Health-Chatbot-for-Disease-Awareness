"""
Offline tests for the ML layer (no API keys needed).

Usage:
    python test_ml.py
"""

import json
import sys
import types
from unittest.mock import MagicMock

# The ML tests never touch RAG/web search, so stub those heavy imports if
# they are not installed (keeps this test runnable without the full stack).
for _m in ["langchain_community", "langchain_community.document_loaders",
           "langchain_community.vectorstores", "langchain.text_splitter",
           "langchain.chains", "tavily"]:
    try:
        __import__(_m)
    except ImportError:
        sys.modules[_m] = MagicMock()

import pipeline
from ml.predictor import SymptomPredictor, SymptomState, Assessment


class StubLLM:
    """Fake LLM: safety classifier + symptom extractor + responder."""

    def __init__(self, extracted):
        self.extracted = extracted  # message text -> list of symptom ids
        self.last_prompt = ""

    def invoke(self, messages):
        system, user = messages[0].content, messages[-1].content
        if "safety classifier" in system:
            return types.SimpleNamespace(
                content="IS_HEALTH: yes\nIS_EMERGENCY: no\nIS_SAFE: yes")
        if "Extract the symptoms" in system:
            return types.SimpleNamespace(content=json.dumps(self.extracted.get(user, [])))
        self.last_prompt = user
        return types.SimpleNamespace(content="stub answer")


def run():
    pred = SymptomPredictor()

    # 1) one symptom -> need more details
    s = SymptomState(symptoms={"high_fever"})
    assert pred.assess(s).status == "need_more"

    # 2) classic dengue pattern -> confident, Dengue first
    s = SymptomState(symptoms={"high_fever", "pain_behind_eyes", "joint_pain", "skin_rash"})
    a = pred.assess(s)
    assert a.status == "confident" and a.top3[0][0] == "Dengue", a

    # 3) generic fever + headache overlaps -> ambiguous with a follow-up
    s = SymptomState(symptoms={"high_fever", "headache"})
    a = pred.assess(s)
    assert a.status == "ambiguous" and a.followup_question, a
    print("fever+headache ->", a.top3, "|", a.followup_question)

    # 4) after MAX_FOLLOWUPS questions we stop asking and say uncertain
    s = SymptomState(symptoms={"high_fever", "headache"}, asked={"a", "b", "c"})
    assert pred.assess(s).status == "uncertain"

    # 5) full pipeline: overlap message -> follow-up appended, then "yes" resolves
    llm = StubLLM({"I have fever and headache": ["high_fever", "headache"]})
    pipeline.rag_agent = lambda q, c: "kb"
    pipeline.search_agent = lambda q, t: "web"
    state = SymptomState()
    out = pipeline.health_chatbot("I have fever and headache", llm, None, None,
                                  predictor=pred, state=state)
    assert "To help narrow this down" in out and state.pending, out
    asked = state.pending
    out2 = pipeline.health_chatbot("yes", llm, None, None, predictor=pred, state=state)
    assert asked in state.symptoms, state
    assert "Symptom-pattern analysis" in llm.last_prompt
    # follow-up replies continue the chat instead of re-answering from scratch
    assert "You previously asked the user" in llm.last_prompt
    assert "Do NOT repeat advice" in llm.last_prompt
    print("after 'yes' to", asked, "->", pred.assess(state).top3)

    # 6) no ML objects -> original behaviour, no ML block in the prompt
    llm2 = StubLLM({})
    pipeline.health_chatbot("What are symptoms of dengue?", llm2, None, None)
    assert "Symptom-pattern" not in llm2.last_prompt

    # 7) deterministic red flag works even when the LLM classifier says "no"
    from agents.safety import has_red_flag
    assert has_red_flag("I have chest pain") and not has_red_flag("I have a cold")

    print("All ML tests passed.")


if __name__ == "__main__":
    run()
