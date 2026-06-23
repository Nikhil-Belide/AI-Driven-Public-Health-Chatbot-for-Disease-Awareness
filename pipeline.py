"""
Pipeline — orchestrates the full multi-agent flow:

  1. Safety Agent   -> checks if message is health-related / emergency / safe
  2. RAG Agent      -> searches the static health knowledge base
  3. Search Agent   -> searches the web for latest related health info
  4. Responder Agent-> combines everything into the final answer

This is the single entry point used by both the CLI test script
and the Streamlit app.
"""

from agents.rag_agent import rag_agent
from agents.search_agent import search_agent
from agents.safety import safety_agent
from agents.responder import generate_response, OUT_OF_SCOPE_MESSAGE


def health_chatbot(question: str, llm, tavily_client, qa_chain) -> str:
    """
    Runs the full multi-agent pipeline for a single user question
    and returns the final response string.
    """
    # Step 1: Safety check
    safety = safety_agent(question, llm)

    # Step 2: If not health related, politely redirect
    if not safety["is_health"]:
        return OUT_OF_SCOPE_MESSAGE

    # Step 3: If unsafe request, refuse
    if not safety["is_safe"]:
        return ("I can't help with that request. If you or someone else is in danger, "
                "please contact a trusted adult, local authorities, or a helpline immediately.")

    # Step 4: RAG search (static knowledge base)
    rag_info = rag_agent(question, qa_chain)

    # Step 5: Web search (latest information)
    web_info = search_agent(question, tavily_client)

    # Step 6: Generate final combined response
    response = generate_response(
        question=question,
        rag_info=rag_info,
        web_info=web_info,
        is_emergency=safety["is_emergency"],
        llm=llm
    )

    return response
