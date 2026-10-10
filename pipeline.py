"""
Pipeline — orchestrates the full multi-agent flow:

  1. Safety Agent   -> checks if message is health-related / emergency / safe
  2. ML Agent       -> extracts symptoms, ranks likely conditions, handles overlap
  3. RAG Agent      -> searches the static health knowledge base
  4. Search Agent   -> searches the web for latest related health info
  5. Responder Agent-> combines everything into the final answer

This is the single entry point used by both the CLI test script
and the Streamlit app.

The ML step is optional: pass predictor=None (default) to get the original
behaviour. For multi-turn follow-up questions, pass a SymptomState that lives
as long as the conversation (e.g. in st.session_state).
"""

from agents.rag_agent import rag_agent
from agents.search_agent import search_agent
from agents.safety import safety_agent
from agents.responder import generate_response, OUT_OF_SCOPE_MESSAGE
from ml.symptoms import SYMPTOM_LABELS
from ml.predictor import (
    SymptomPredictor, SymptomState, update_state_from_message,
    format_for_llm, _YES, _NO,
)


def health_chatbot(question: str, llm, tavily_client, qa_chain,
                   predictor: SymptomPredictor = None,
                   state: SymptomState = None) -> str:
    """
    Runs the full multi-agent pipeline for a single user question
    and returns the final response string.
    """
    use_ml = predictor is not None and state is not None

    # A bare yes/no replying to our follow-up question is not a standalone
    # message, so it skips the scope check and is rewritten for retrieval.
    is_followup_reply = (
        use_ml and state.pending
        and question.strip().lower().rstrip(".!") in (_YES | _NO)
    )

    # Step 1: Safety check
    if is_followup_reply:
        safety = {"is_health": True, "is_safe": True, "is_emergency": False}
    else:
        safety = safety_agent(question, llm)

    # Step 2: If not health related, politely redirect
    if not safety["is_health"]:
        return OUT_OF_SCOPE_MESSAGE

    # Step 3: If unsafe request, refuse
    if not safety["is_safe"]:
        return ("I can't help with that request. If you or someone else is in danger, "
                "please contact a trusted adult, local authorities, or a helpline immediately.")

    # Step 4: ML symptom assessment (advisory only)
    ml_info, followup = "", None
    search_question = question
    if use_ml:
        update_state_from_message(question, state, llm, predictor)
        assessment = predictor.assess(state)
        ml_info = format_for_llm(assessment)
        if assessment.status == "ambiguous":
            followup = assessment.followup_question
            state.pending = assessment.followup_symptom
        if is_followup_reply and state.symptoms:
            names = [SYMPTOM_LABELS.get(s, s) for s in sorted(state.symptoms)]
            search_question = (f"I have: {', '.join(names)}. "
                               "What could this mean and what should I do?")

    # Step 5: RAG search (static knowledge base)
    rag_info = rag_agent(search_question, qa_chain)

    # Step 6: Web search (latest information)
    web_info = search_agent(search_question, tavily_client)

    # Step 7: Generate final combined response
    response = generate_response(
        question=search_question,
        rag_info=rag_info,
        web_info=web_info,
        is_emergency=safety["is_emergency"],
        llm=llm,
        ml_info=ml_info,
    )

    if followup:
        response += (f"\n\n**To help narrow this down:** {followup} "
                     "(reply *yes* or *no*, or tell me anything else you're feeling)")

    return response
