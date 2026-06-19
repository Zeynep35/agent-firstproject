from logger_config import logger
from agent_core import get_agent, stream_llm_response, get_llm
from router import router
from rag import create_vectorstore_from_pdfs, load_existing_vectorstore

import streamlit as st
import os
import time


agent = get_agent()
llm = get_llm()

st.title("AgentDemo + Çoklu PDF RAG")

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
    try:
        st.session_state.vectorstore = load_existing_vectorstore()
    except Exception:
        logger.exception("Mevcut Chroma veritabanı yüklenemedi.")
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


uploaded_files = st.file_uploader(
    "PDF yükle",
    accept_multiple_files=True,
    type=["pdf"]
)

if uploaded_files:
    st.info(f"{len(uploaded_files)} PDF seçildi.")

    if st.button("PDF'leri işle"):
        try:
            file_names = [file.name for file in uploaded_files]
            logger.info(f"PDF'ler yüklendi: {file_names}")

            vectorstore, message = create_vectorstore_from_pdfs(uploaded_files)

            st.session_state.vectorstore = vectorstore

            logger.info(message)
            st.success(message)

        except Exception:
            logger.exception("PDF'ler işlenirken hata oluştu.")
            st.error("PDF'ler işlenirken hata oluştu. Detaylar agent.log dosyasına kaydedildi.")


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

    assistant_answer = ""

    try:
        quick_answer = router(
            user_input,
            vectorstore=st.session_state.vectorstore,
            llm = llm
        )

        if quick_answer is not None:
            assistant_answer = quick_answer

            with st.chat_message("assistant"):
                st.write(assistant_answer)

        else:
            with st.chat_message("assistant"):
                placeholder = st.empty()

                for token in stream_llm_response(user_input):
                    assistant_answer += token
                    placeholder.write(assistant_answer)

        elapsed_time = time.time() - start_time

        logger.info(f"Asistan cevabı: {assistant_answer}")
        logger.info(f"Cevap süresi: {elapsed_time:.2f} saniye")

    except Exception:
        logger.exception("Cevap üretilirken hata oluştu.")
        assistant_answer = "Bir hata oluştu. Detaylar agent.log dosyasına kaydedildi."

        with st.chat_message("assistant"):
            st.write(assistant_answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_answer}
    )


if st.button("Hafızayı göster"):
    state = agent.get_state(config)
    st.write(state.values)