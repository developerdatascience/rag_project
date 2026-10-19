
import uuid

import streamlit as st

from ui.api_client import RAGAPIClient
from ui.components import render_sources


st.set_page_config(
    page_title="Document RAG Chat",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Document RAG Chat")
st.caption(
    "Ask questions about your indexed documents. "
    "Answers depend on the evidence retrieved by the backend."
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "top_k" not in st.session_state:
    st.session_state.top_k = 3

client = RAGAPIClient()

with st.sidebar:
    st.header("Settings")

    st.slider(
        "Retrieved chunks (top_k)",
        min_value=1,
        max_value=10,
        key="top_k",
    )

    if client.health_check():
        st.success("RAG API connected")
    else:
        st.error("RAG API unavailable")
        st.caption(
            "Start the FastAPI server and check RAG_API_URL."
        )

    st.caption("Session")
    st.code(st.session_state.session_id)

    if st.button("Start a new conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# Display messages from this Streamlit session.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

question = st.chat_input("Ask a question about your documents")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()

        try:
            answer = st.write_stream(
                client.stream_chat(
                    question=question,
                    session_id=st.session_state.session_id,
                    top_k=st.session_state.top_k,
                )
            )

            if not isinstance(answer, str):
                answer = str(answer)

            if not answer.strip():
                raise RuntimeError(
                    "The API completed without returning an answer."
                )

            answer_placeholder.markdown(answer)

            sources = client.last_sources
            render_sources(sources)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception:
            st.error(
                "The answer could not be completed. "
                "Check the API logs and try again."
            )
