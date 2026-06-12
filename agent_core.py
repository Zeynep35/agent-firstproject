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