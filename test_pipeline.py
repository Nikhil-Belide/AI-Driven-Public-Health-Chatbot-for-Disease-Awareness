"""
Quick CLI test script — lets you test the pipeline directly
in a terminal or notebook without running Streamlit.

Usage:
    python test_pipeline.py
"""

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from tavily import TavilyClient

from agents.rag_agent import build_rag_chain
from pipeline import health_chatbot
from ml.predictor import SymptomPredictor, SymptomState

# ── Paste your API keys here for quick testing ──
GROQ_API_KEY = "PASTE_YOUR_GROQ_KEY_HERE"
TAVILY_API_KEY = "PASTE_YOUR_TAVILY_KEY_HERE"


def main():
    print("Loading models... (first run takes ~30s)")
    llm = ChatGroq(api_key=GROQ_API_KEY, model="qwen/qwen3-32b")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    qa_chain = build_rag_chain(llm, embeddings, kb_path="data/health_kb.txt")
    predictor = SymptomPredictor()
    state = SymptomState()
    print("✅ Ready!\n")

    test_questions = [
        "What are symptoms of dengue fever?",
        "How can I prevent malaria?",
        "I am having severe chest pain and difficulty breathing!",
        "What is the capital of France?",
        "I have high fever, headache and pain behind my eyes with joint pain",
        "I have a fever and a headache",   # overlapping -> expect a follow-up question
    ]

    for q in test_questions:
        print("=" * 70)
        print(f"Q: {q}")
        answer = health_chatbot(q, llm, tavily_client, qa_chain,
                                predictor=predictor, state=state)
        print(f"A: {answer}\n")


if __name__ == "__main__":
    main()
