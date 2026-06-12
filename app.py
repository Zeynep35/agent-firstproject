from logger_config import logger
from agent_core import get_agent
from rag import create_vectorstore
from router import router

import streamlit as st
import os
import time


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