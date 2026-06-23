"""
Responder Agent — combines information from the RAG agent
(static knowledge base) and the Search agent (live web results)
into one clear, empathetic, safe final answer for the user.
"""

from langchain_core.messages import SystemMessage, HumanMessage

RESPONSE_SYSTEM_PROMPT = """You are a helpful public health assistant chatbot.

Guidelines:
- Provide accurate, clear, and easy-to-understand health information.
- Use simple language suitable for the general public, not medical jargon.
- Be empathetic and supportive in tone.
- Structure your answer with short paragraphs or bullet points where helpful.
- Never provide a definitive diagnosis — only general information.
- Always end your response with a reminder to consult a qualified doctor.
"""

EMERGENCY_BANNER = (
    "\n🚨 **THIS MAY BE A MEDICAL EMERGENCY** 🚨\n"
    "📞 Call 108 (Ambulance) immediately\n"
    "🏥 Go to the nearest hospital / emergency room now\n"
    "Do not wait — act immediately.\n\n"
)


def generate_response(question: str, rag_info: str, web_info: str,
                       is_emergency: bool, llm) -> str:
    """
    Builds the final chatbot response using RAG context,
    live web search context, and an LLM call, then prepends
    an emergency banner if needed.
    """
    user_prompt = f"""Question: {question}

Relevant information from health knowledge base:
{rag_info}

Relevant recent information from the web:
{web_info}

Using the information above, write a helpful and safe response
to the user's question. End with a reminder to consult a doctor
for personal medical advice."""

    response = llm.invoke([
        SystemMessage(content=RESPONSE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    final_text = response.content
    if "</think>" in final_text:
        final_text = final_text.split("</think>")[-1].strip()

    if is_emergency:
        final_text = EMERGENCY_BANNER + final_text

    return final_text


OUT_OF_SCOPE_MESSAGE = """I'm a Public Health Chatbot and can only help with health-related questions! 🏥

You can ask me about:
• Disease symptoms and causes
• Prevention tips
• General treatment information
• When to seek emergency care
• Health helpline numbers

How can I help with your health today?"""
