import sqlite3
import streamlit as st

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver


@st.cache_resource
def get_llm():
    return ChatOllama(
        model="mistral:latest",
        num_gpu=0,
        temperature=0.1
    )


@st.cache_resource
def get_agent():
    llm = get_llm()

    conn = sqlite3.connect("memory.sqlite", check_same_thread=False)

    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

    agent = create_agent(
        model=llm,
        tools=[],
        system_prompt="""
Sen konuşma geçmişini dikkatlice kullanan bir asistansın.

Kurallar:
- Kullanıcı geçmişte kendisi hakkında bilgi verdiyse onu hatırla.
- Kısa ve net Türkçe cevap ver.
- Kendini kullanıcıyla karıştırma.
""",
        checkpointer=checkpointer,
    )

    return agent


def stream_llm_response(user_input: str, config=None):
    """
    Agent üzerinden cevap üretir.
    Böylece SQLite checkpointer / thread_id hafızası çalışır.
    """

    agent = get_agent()

    if config is None:
        config = {
            "configurable": {
                "thread_id": "default"
            }
        }

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        },
        config=config
    )

    messages = result.get("messages", [])

    if not messages:
        yield "Cevap üretilemedi."
        return

    last_message = messages[-1]
    content = getattr(last_message, "content", str(last_message))

    yield content