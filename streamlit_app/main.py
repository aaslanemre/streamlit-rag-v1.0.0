import streamlit as st
import requests
import json

# ── Configuration ────────────────────────────────────────────────────────────
N8N_WEBHOOK_URL = "http://localhost:5678/webhook/rag-chat"

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🤖 RAG Chatbot")
st.caption("Powered by Llama 3.2 · n8n · Ollama")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask something…"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send to n8n webhook
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = requests.post(
                    N8N_WEBHOOK_URL,
                    json={
                        "message": prompt,
                        "session_id": st.session_state.get("session_id", "default"),
                    },
                    timeout=60,
                )
                response.raise_for_status()

                data = response.json()
                # n8n AI Agent node returns the answer under the "output" key
                answer = (
                    data.get("output")
                    or data.get("text")
                    or data.get("response")
                    or str(data)
                )
            except requests.exceptions.ConnectionError:
                answer = (
                    "⚠️ Cannot reach n8n. "
                    "Make sure the Docker stack is running (`docker compose up -d`)."
                )
            except requests.exceptions.Timeout:
                answer = "⚠️ The request timed out. The model may still be loading."
            except Exception as e:
                answer = f"⚠️ Unexpected error: {e}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
