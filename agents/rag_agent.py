"""
RAG Agent — searches the health knowledge base
using semantic search (embeddings + ChromaDB).
"""

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

_qa_chain_cache = {}


def build_rag_chain(llm, embeddings, kb_path: str = "data/health_kb.txt"):
    """
    Builds (or returns cached) RAG chain over the health knowledge base.

    Steps:
    1. Load the text file
    2. Split into overlapping chunks
    3. Embed and store chunks in ChromaDB
    4. Wrap as a retriever + RetrievalQA chain
    """
    cache_key = kb_path
    if cache_key in _qa_chain_cache:
        return _qa_chain_cache[cache_key]

    loader = TextLoader(kb_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    _qa_chain_cache[cache_key] = qa_chain
    return qa_chain


def rag_agent(question: str, qa_chain) -> str:
    """
    Searches the health knowledge base for an answer
    relevant to the user's question.
    """
    try:
        result = qa_chain.invoke({"query": question})
        return result["result"]
    except Exception as e:
        return f"Could not search knowledge base: {str(e)}"
