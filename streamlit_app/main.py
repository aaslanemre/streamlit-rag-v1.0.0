import streamlit as st
import requests

# ── Configuração ──────────────────────────────────────────────────────────────
N8N_WEBHOOK_URL = "http://localhost:5678/webhook/rag-chat"

st.set_page_config(
    page_title="Sistema de Análise Documental",
    page_icon="📄",
    layout="centered",
)

# ── Estado da sessão ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Interface ─────────────────────────────────────────────────────────────────
st.caption("Sistema de análise documental ativo. Insira sua consulta técnica abaixo.")

# Histórico de mensagens
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada de consulta
if prompt := st.chat_input("Digite sua pergunta técnica..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando base de dados técnica..."):
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
                answer = (
                    data.get("output")
                    or data.get("text")
                    or data.get("response")
                    or str(data)
                )
            except requests.exceptions.ConnectionError:
                answer = "Erro de conexão com o servidor de processamento."
            except requests.exceptions.Timeout:
                answer = "A solicitação excedeu o tempo limite. O modelo pode estar sendo inicializado."
            except Exception as e:
                answer = f"Erro inesperado: {e}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
