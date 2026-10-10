"""
AI-Driven Public Health Chatbot — Streamlit App

Run with:
    streamlit run app.py
"""

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from tavily import TavilyClient

from agents.rag_agent import build_rag_chain
from pipeline import health_chatbot
from ml.predictor import SymptomPredictor, SymptomState

# ── Page config ──
st.set_page_config(
    page_title="AI Public Health Chatbot",
    page_icon="🏥",
    layout="wide"
)

# ── Header ──
st.markdown("""
<div style="background: linear-gradient(90deg, #1a1a2e, #16213e);
     padding: 24px; border-radius: 12px; margin-bottom: 24px;">
    <h1 style="color: white; text-align: center; margin: 0;">
        🏥 AI-Driven Public Health Chatbot
    </h1>
    <p style="color: #b0b0c0; text-align: center; margin-top: 8px;">
        Disease Awareness · Prevention Tips · Emergency Guidance
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Settings")
    groq_key = st.text_input("Groq API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password")

    st.divider()
    st.header("🔍 Quick Topics")
    quick_topics = [
        "Dengue symptoms",
        "Malaria prevention",
        "COVID-19 information",
        "Diabetes management",
        "Emergency helpline numbers"
    ]
    for topic in quick_topics:
        if st.button(topic, use_container_width=True):
            st.session_state.quick_question = topic

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.symptom_state = SymptomState()
        st.rerun()

    st.divider()
    st.warning(
        "⚠️ **Disclaimer**\n\n"
        "This chatbot provides general health information only. "
        "It is not a substitute for professional medical advice. "
        "Always consult a qualified doctor."
    )

# ── Resource initialization (cached) ──
@st.cache_resource(show_spinner=False)
def load_resources(groq_api_key: str):
    llm = ChatGroq(api_key=groq_api_key, model="qwen/qwen3-32b")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    qa_chain = build_rag_chain(llm, embeddings, kb_path="data/health_kb.txt")
    predictor = SymptomPredictor()  # trains a starter model on first run
    return llm, qa_chain, predictor


# ── Chat state ──
if "symptom_state" not in st.session_state:
    st.session_state.symptom_state = SymptomState()
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "👋 Hello! I am your AI Public Health Assistant.\n\n"
            "I can help you with:\n"
            "- 🤒 Disease symptoms and causes\n"
            "- 🛡️ Prevention tips\n"
            "- 💊 General treatment information\n"
            "- 🚑 When to seek emergency care\n"
            "- 📞 Health helpline numbers\n\n"
            "How can I help you today?"
        )
    }]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def handle_question(question: str):
    """Runs the full pipeline for a question and updates chat history."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not groq_key or not tavily_key:
            answer = "⚠️ Please enter your Groq and Tavily API keys in the sidebar first!"
            st.markdown(answer)
        else:
            with st.spinner("🔍 Searching health knowledge base and latest news..."):
                try:
                    llm, qa_chain, predictor = load_resources(groq_key)
                    tavily_client = TavilyClient(api_key=tavily_key)
                    answer = health_chatbot(
                        question, llm, tavily_client, qa_chain,
                        predictor=predictor,
                        state=st.session_state.symptom_state,
                    )
                except Exception as e:
                    answer = f"Something went wrong: {str(e)}"
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


# Handle quick topic button clicks
if "quick_question" in st.session_state:
    q = st.session_state.quick_question
    del st.session_state.quick_question
    handle_question(q)

# Chat input box
if user_question := st.chat_input("Ask a health question..."):
    handle_question(user_question)
