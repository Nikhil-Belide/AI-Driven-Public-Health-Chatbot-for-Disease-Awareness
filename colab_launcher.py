# ============================================================
# AI-Driven Public Health Chatbot — Google Colab Launcher
# ============================================================
# Copy each "CELL" block below into a separate Colab cell,
# in order, and run them top to bottom.
# ============================================================

# ── CELL 1 — Install dependencies ──
"""
!pip install streamlit langchain langchain-groq langchain-community chromadb sentence-transformers tavily-python pyngrok -q
"""

# ── CELL 2 — Upload project files ──
"""
# Upload the HealthChatbot folder to Colab first
# (drag and drop the whole folder into the Colab file browser,
#  or zip it and unzip in Colab with:
#  !unzip HealthChatbot.zip
#  %cd HealthChatbot
"""

# ── CELL 3 — Set your API keys as environment variables (optional) ──
"""
import os
os.environ["GROQ_API_KEY"] = "PASTE_YOUR_GROQ_KEY_HERE"
os.environ["TAVILY_API_KEY"] = "PASTE_YOUR_TAVILY_KEY_HERE"
"""

# ── CELL 4 — Quick test without Streamlit ──
"""
!python test_pipeline.py
"""

# ── CELL 5 — Launch Streamlit with a public URL ──
"""
from pyngrok import ngrok
import subprocess, threading, time

ngrok.set_auth_token("PASTE_YOUR_NGROK_TOKEN_HERE")

def run():
    subprocess.run(["streamlit", "run", "app.py",
                     "--server.port", "8502",
                     "--server.headless", "true"])

thread = threading.Thread(target=run, daemon=True)
thread.start()
time.sleep(5)

url = ngrok.connect(8502)
print(f"🌐 Your Health Chatbot is LIVE at: {url}")
"""
