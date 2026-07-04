from logger_config import logger
from agent_core import get_agent, get_llm
from router import router
from agentic_rag import should_use_rag

import streamlit as st
import os
import time
import json
from datetime import datetime

from rag import (
    create_vectorstore_from_pdfs,
    calculate_file_hash,
    load_existing_vectorstore,
    list_indexed_pdfs,
    delete_pdf_from_vectorstore,
    clear_vectorstore,
    stream_rag_answer
)


# =========================
# Agent / LLM
# =========================

agent = get_agent()
llm = get_llm()


# =========================
# Helper Functions
# =========================

def render_stream(generator):
    """
    Gelen stream cevabını Streamlit ekranına canlı canlı yazar.
    """
    placeholder = st.empty()
    full_response = ""

    for token in generator:
        full_response += token
        placeholder.markdown(full_response + "▌")

    placeholder.markdown(full_response)
    return full_response


def stream_llm_answer(user_input, llm):
    """
    Normal LLM cevabını token token stream eder.
    """
    if llm is None:
        yield "LLM yüklenemedi."
        return

    for chunk in llm.stream(user_input):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)

        if token:
            yield token


def filter_duplicate_uploaded_files(uploaded_files):
    """
    Aynı yükleme içinde seçilen duplicate PDF'leri engeller.
    Dosya adına değil, dosya içeriğinin hash'ine bakar.
    """
    if not uploaded_files:
        return [], []

    seen_hashes = set()
    unique_files = []
    duplicate_files = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        file_hash = calculate_file_hash(file_bytes)

        if file_hash in seen_hashes:
            duplicate_files.append(uploaded_file.name)
            continue

        seen_hashes.add(file_hash)
        unique_files.append(uploaded_file)

    return unique_files, duplicate_files


def export_chat_as_txt(messages, thread_id):
    lines = []

    lines.append("Chat Export")
    lines.append(f"Sohbet ID: {thread_id}")
    lines.append(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 50)
    lines.append("")

    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content", "")

        if role == "user":
            role_name = "Kullanıcı"
        elif role == "assistant":
            role_name = "Asistan"
        else:
            role_name = role

        lines.append(f"{role_name}:")
        lines.append(content)
        lines.append("-" * 50)

    return "\n".join(lines)


def export_chat_as_json(messages, thread_id):
    data = {
        "thread_id": thread_id,
        "exported_at": datetime.now().isoformat(),
        "messages": messages
    }

    return json.dumps(data, ensure_ascii=False, indent=2)


# =========================
# Page Config
# =========================

st.title("AgentDemo V2 - Multi PDF Agentic RAG")


# =========================
# Session State
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    try:
        st.session_state.vectorstore = load_existing_vectorstore()
    except Exception:
        logger.exception("Mevcut Chroma veritabanı yüklenemedi.")
        st.session_state.vectorstore = None


# =========================
# Sidebar
# =========================

with st.sidebar:
    st.title("Ayarlar")

    thread_id = st.text_input(
        "Sohbet ID",
        value="zeynep_1"
    )

    st.info(
        "Her farklı Sohbet ID ayrı hafıza kullanır. "
        "Örnek: zeynep_1, test_1, pdf_1"
    )

    if st.button("Ekran geçmişini temizle", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("Logları göster", use_container_width=True):
        if os.path.exists("agent.log"):
            with open("agent.log", "r", encoding="utf-8") as f:
                st.text(f.read()[-3000:])
        else:
            st.warning("Henüz log dosyası yok.")

    st.divider()

    # =========================
    # PDF Yönetimi
    # =========================

    st.header("PDF Yönetimi")

    if st.session_state.vectorstore is None:
        try:
            st.session_state.vectorstore = load_existing_vectorstore()
        except Exception:
            logger.exception("PDF veritabanı yüklenemedi.")
            st.error("PDF veritabanı yüklenemedi.")

    indexed_pdfs = list_indexed_pdfs(st.session_state.vectorstore)

    if indexed_pdfs:
        st.caption(f"{len(indexed_pdfs)} PDF veritabanında kayıtlı.")

        selected_pdf_to_delete = st.selectbox(
            "Kayıtlı PDF seç",
            indexed_pdfs
        )

        if st.button("Seçili PDF'i Sil", use_container_width=True):
            st.session_state.vectorstore, delete_message = delete_pdf_from_vectorstore(
                st.session_state.vectorstore,
                selected_pdf_to_delete
            )
            st.success(delete_message)
            st.rerun()

        st.divider()

        confirm_clear = st.checkbox("Tüm PDF verilerini silmeyi onaylıyorum")

        if st.button(
            "Tüm PDF Verilerini Temizle",
            use_container_width=True,
            disabled=not confirm_clear
        ):
            st.session_state.vectorstore, clear_message = clear_vectorstore(
                st.session_state.vectorstore
            )
            st.warning(clear_message)
            st.rerun()

    else:
        st.info("Henüz kayıtlı PDF yok.")

    st.divider()

    # =========================
    # Export
    # =========================

    if st.session_state.messages:
        txt_data = export_chat_as_txt(
            st.session_state.messages,
            thread_id
        )

        json_data = export_chat_as_json(
            st.session_state.messages,
            thread_id
        )

        st.download_button(
            label="Chat'i TXT indir",
            data=txt_data,
            file_name=f"chat_export_{thread_id}.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.download_button(
            label="Chat'i JSON indir",
            data=json_data,
            file_name=f"chat_export_{thread_id}.json",
            mime="application/json",
            use_container_width=True
        )
    else:
        st.info("Export için henüz mesaj yok.")

    st.divider()

    # =========================
    # Debug
    # =========================

    with st.expander("Hafıza / Debug"):
        if st.button(
            "Hafızayı göster",
            key="show_memory_sidebar",
            use_container_width=True
        ):
            st.write(st.session_state)


# =========================
# Config
# =========================

config = {
    "configurable": {
        "thread_id": thread_id
    }
}


# =========================
# Process Message
# =========================

if "process_message" in st.session_state:
    st.success(st.session_state.process_message)
    del st.session_state.process_message


# =========================
# PDF Upload
# =========================

uploaded_files = st.file_uploader(
    "PDF yükle",
    accept_multiple_files=True,
    type=["pdf"]
)

use_vision = st.checkbox(
    "Vision analizi de yap",
    value=False,
    help="PDF sayfalarını görsel olarak yorumlar. Daha yavaş çalışır."
)

max_vision_pages = st.number_input(
    "Vision ile okunacak maksimum sayfa sayısı",
    min_value=1,
    max_value=10,
    value=3
)

unique_uploaded_files, duplicate_uploaded_files = filter_duplicate_uploaded_files(
    uploaded_files
)

if uploaded_files:
    st.info(
        f"{len(uploaded_files)} PDF seçildi. "
        f"Benzersiz PDF sayısı: {len(unique_uploaded_files)}"
    )

    if duplicate_uploaded_files:
        st.warning(
            "Duplicate olduğu için atlanacak PDF'ler: "
            + ", ".join(duplicate_uploaded_files)
        )

    if st.button("PDF'leri işle"):
        if not unique_uploaded_files:
            st.warning("İşlenecek yeni PDF yok. Seçilen dosyalar duplicate olabilir.")
        else:
            try:
                with st.spinner("PDF'ler işleniyor..."):
                    st.session_state.vectorstore, message = create_vectorstore_from_pdfs(
                        unique_uploaded_files,
                        use_vision=use_vision,
                        max_vision_pages=int(max_vision_pages)
                    )

                st.session_state.process_message = message
                st.rerun()

            except Exception:
                logger.exception("PDF'ler işlenirken hata oluştu.")
                st.error(
                    "PDF'ler işlenirken hata oluştu. "
                    "Detaylar agent.log dosyasına kaydedildi."
                )


# =========================
# Chat History
# =========================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# =========================
# Chat Input
# =========================

user_input = st.chat_input("Mesaj yaz...")

if user_input:
    start_time = time.time()

    logger.info(f"Kullanıcı mesajı: {user_input}")
    logger.info(f"Thread ID: {thread_id}")

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.write(user_input)

    assistant_answer = ""

    try:
        with st.chat_message("assistant"):

            # 1. PDF sorusu ise gerçek streaming RAG çalışır.
            if (
                st.session_state.vectorstore is not None
                and should_use_rag(user_input)
            ):
                assistant_answer = render_stream(
                    stream_rag_answer(
                        question=user_input,
                        vectorstore=st.session_state.vectorstore,
                        llm=llm
                    )
                )

            else:
                # 2. Router hızlı cevap verebiliyorsa onu kullanır.
                quick_answer = router(
                    user_input,
                    vectorstore=st.session_state.vectorstore,
                    llm=llm
                )

                if quick_answer:
                    assistant_answer = str(quick_answer)
                    st.markdown(assistant_answer)

                else:
                    # 3. Router cevap vermezse normal LLM streaming çalışır.
                    assistant_answer = render_stream(
                        stream_llm_answer(
                            user_input,
                            llm
                        )
                    )

    except Exception as e:
        logger.exception("Cevap üretilirken hata oluştu.")
        assistant_answer = (
            f"Cevap üretilirken hata oluştu: {e}\n\n"
            "Detaylar agent.log dosyasına kaydedildi."
        )
        st.error(assistant_answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_answer
        }
    )

    elapsed_time = round(time.time() - start_time, 2)
    logger.info(f"Cevap süresi: {elapsed_time} saniye")


# =========================
# Agent Memory Main Button
# =========================

if st.button("Agent hafızasını göster", key="show_agent_memory_main"):
    try:
        state = agent.get_state(config)
        st.write(state.values)
    except Exception:
        logger.exception("Agent hafızası gösterilirken hata oluştu.")
        st.error("Agent hafızası gösterilirken hata oluştu.")