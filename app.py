from logger_config import logger
from tools import *

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver

import sqlite3
import streamlit as st
import os
import logging
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# =====================
# LLM & AGENT
# =====================

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


# =====================
# RAG
# =====================

@st.cache_resource
def create_vectorstore(pdf_path: str):
    logger.info(f"Vectorstore oluşturuluyor: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    logger.info(f"PDF sayfa sayısı: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    logger.info(f"Chunk sayısı: {len(chunks)}")

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    logger.info("Vectorstore başarıyla oluşturuldu.")

    return vectorstore


def ask_rag(question: str, vectorstore):
    logger.info(f"RAG sorusu alındı: {question}")

    docs = vectorstore.similarity_search(question, k=3)

    logger.info(f"RAG için {len(docs)} kaynak bulundu.")

    context = ""

    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "Bilinmiyor")

        context += f"""
[KAYNAK {i}]
Sayfa: {page + 1 if isinstance(page, int) else page}
İçerik:
{doc.page_content}
"""

    llm = get_llm()

    prompt = f"""
Sen Türkçe cevap veren bir RAG asistanısın.

Aşağıdaki belge parçalarını kullanarak soruyu cevapla.
Cevabın sonunda hangi kaynakları kullandığını belirt.

Belge parçaları:
{context}

Kullanıcı sorusu:
{question}

Cevap formatı:

Cevap:
...

Kaynaklar:
- Kaynak 1, Sayfa ...
- Kaynak 2, Sayfa ...

Eğer cevap belgede yoksa:
"Bu bilgi belgede bulunmuyor." de.
"""

    response = llm.invoke(prompt)

    return response.content


# =====================
# ROUTER
# =====================

def router(user_input: str, vectorstore=None):
    text = user_input.lower()

    if vectorstore is not None and (
        "pdf" in text
        or "belge" in text
        or "doküman" in text
        or "dosya" in text
        or "bu metinde" in text
        or "bu belgede" in text
    ):
        logger.info("Router: RAG seçildi.")
        return ask_rag(user_input, vectorstore)

    if "hava" in text:
        logger.info("Router: Hava durumu seçildi.")

        if "izmir" in text:
            return "İzmir'de hava güneşli."
        elif "istanbul" in text:
            return "İstanbul için hava bulutlu."
        else:
            return "Hangi şehir için hava durumunu istiyorsun?"

    if "eksi" in text or "çıkar" in text:
        logger.info("Router: Çıkarma işlemi seçildi.")
        return "Çıkarma işlemi için örnek: 10'dan 3 çıkar."

    if "internet" in text or "web" in text or "ara" in text:
        logger.info("Router: Web search demo seçildi.")
        return "Web arama şu an demo modunda: Bulunan sonuçlar..."

    logger.info("Router: Agent seçildi.")
    return None


# =====================
# STREAMLIT UI
# =====================

agent = get_agent()

st.title("AgentDemo + RAG")

st.sidebar.title("Ayarlar")

thread_id = st.sidebar.text_input(
    "Sohbet ID",
    value="zeynep_1"
)

st.sidebar.info(
    "Her farklı Sohbet ID ayrı hafıza kullanır. Örnek: zeynep_1, test_1, pdf_1"
)

config = {
    "configurable": {
        "thread_id": thread_id
    }
}

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


if st.sidebar.button("Ekran geçmişini temizle"):
    st.session_state.messages = []
    st.rerun()


if st.sidebar.button("Logları göster"):
    if os.path.exists("agent.log"):
        with open("agent.log", "r", encoding="utf-8") as f:
            st.sidebar.text(f.read()[-3000:])
    else:
        st.sidebar.warning("Henüz log dosyası yok.")


uploaded_file = st.file_uploader("PDF yükle", type=["pdf"])

if uploaded_file is not None:
    try:
        logger.info(f"PDF yüklendi: {uploaded_file.name}")

        os.makedirs("uploads", exist_ok=True)

        pdf_path = os.path.join("uploads", uploaded_file.name)

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.vectorstore = create_vectorstore(pdf_path)

        logger.info(f"PDF vektör veritabanına işlendi: {uploaded_file.name}")

        st.success("PDF işlendi. Artık belge hakkında soru sorabilirsin.")

    except Exception:
        logger.exception("PDF işlenirken hata oluştu.")
        st.error("PDF işlenirken hata oluştu. Detaylar agent.log dosyasına kaydedildi.")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


user_input = st.chat_input("Mesaj yaz...")

if user_input:
    start_time = time.time()

    logger.info(f"Kullanıcı mesajı: {user_input}")
    logger.info(f"Thread ID: {thread_id}")

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    try:
        quick_answer = router(
            user_input,
            vectorstore=st.session_state.vectorstore
        )

        if quick_answer is not None:
            assistant_answer = quick_answer
        else:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config
            )

            assistant_answer = result["messages"][-1].content

        elapsed_time = time.time() - start_time

        logger.info(f"Asistan cevabı: {assistant_answer}")
        logger.info(f"Cevap süresi: {elapsed_time:.2f} saniye")

    except Exception:
        logger.exception("Cevap üretilirken hata oluştu.")
        assistant_answer = "Bir hata oluştu. Detaylar agent.log dosyasına kaydedildi."

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_answer}
    )

    with st.chat_message("assistant"):
        st.write(assistant_answer)


if st.button("Hafızayı göster"):
    state = agent.get_state(config)
    st.write(state.values)