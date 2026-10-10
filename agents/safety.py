"""
Safety Agent — classifies each incoming user message before
it is processed further. This is a critical guardrail for any
health-related chatbot.

It checks:
1. Is the message actually health-related?
2. Does it describe a possible medical emergency?
3. Is it safe to respond to (no harmful/illegal request)?
"""

from langchain_core.messages import SystemMessage, HumanMessage

SAFETY_PROMPT = """You are a safety classifier for a public health chatbot.
Analyze the user's message and respond with EXACTLY this format, nothing else:

IS_HEALTH: yes/no
IS_EMERGENCY: yes/no
IS_SAFE: yes/no

Rules:
- IS_HEALTH = yes if the message is about symptoms, diseases, prevention,
  treatment, wellness, or medical helplines.
- IS_EMERGENCY = yes if the message describes severe symptoms such as
  chest pain, difficulty breathing, heavy bleeding, loss of consciousness,
  severe allergic reaction, or says "emergency"/"urgent"/"dying".
- IS_SAFE = no only if the user is asking for something harmful
  (e.g. how to harm themselves or others, illegal drug synthesis, etc).
  Otherwise IS_SAFE = yes.
"""


RED_FLAG_PHRASES = [
    "chest pain", "chest pressure", "can't breathe", "cannot breathe",
    "difficulty breathing", "trouble breathing", "struggling to breathe",
    "coughing blood", "coughing up blood", "vomiting blood", "heavy bleeding",
    "unconscious", "fainted", "seizure", "convulsion", "stroke",
    "face drooping", "slurred speech", "bluish lips", "overdose",
    "suicid", "kill myself", "want to die",
]


def has_red_flag(text: str) -> bool:
    """Deterministic emergency check. Runs in addition to the LLM classifier
    so a classifier miss can never suppress the emergency banner."""
    t = text.lower()
    return any(p in t for p in RED_FLAG_PHRASES)


def safety_agent(question: str, llm) -> dict:
    """
    Returns a dict with three boolean flags describing the message:
    {"is_health": bool, "is_emergency": bool, "is_safe": bool}
    """
    response = llm.invoke([
        SystemMessage(content=SAFETY_PROMPT),
        HumanMessage(content=question)
    ])

    text = response.content.lower()

    return {
        "is_health": "is_health: yes" in text,
        "is_emergency": "is_emergency: yes" in text or has_red_flag(question),
        "is_safe": "is_safe: no" not in text
    }
