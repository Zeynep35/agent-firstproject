import sqlite3
import streamlit as st
import os

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from logger_config import logger

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def get_installed_ollama_models():
    """
    Ollama'da yüklü modelleri okur.
    Hata olursa boş liste döner.
    """
    try:
        import ollama

        models_response = ollama.list()
        models = models_response.get("models", [])

        model_names = []

        for model in models:
            name = model.get("name") or model.get("model")
            if name:
                model_names.append(name)

        return model_names

    except Exception:
        logger.warning("Ollama model listesi okunamadı.", exc_info=True)
        return []

def pick_available_text_model():
    """
    .env içindeki ana modeli ve fallback modelleri sırayla kontrol eder.
    Ollama'da yüklü olan ilk modeli seçer.
    """

    primary_model = os.getenv("TEXT_MODEL", "qwen2.5:1.5b")

    fallback_models = os.getenv(
        "TEXT_MODEL_FALLBACKS",
        "llama3.2:1b,gemma2:2b,mistral:latest"
    )

    candidate_models = [primary_model]

    for model_name in fallback_models.split(","):
        model_name = model_name.strip()
        if model_name and model_name not in candidate_models:
            candidate_models.append(model_name)

    installed_models = get_installed_ollama_models()

    if not installed_models:
        logger.warning(
            "Yüklü Ollama modelleri okunamadı. Ana model kullanılacak: %s",
            primary_model
        )
        return primary_model

    for model_name in candidate_models:
        if model_name in installed_models:
            logger.info("Seçilen text model: %s", model_name)
            return model_name

    logger.warning(
        "Aday modellerin hiçbiri yüklü değil. Ana model deneniyor: %s",
        primary_model
    )

    return primary_model

@st.cache_resource
def get_llm():
    try:
        model_name = pick_available_text_model()

        ollama_base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434"
        )

        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "1536"))
        num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "256"))
        keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "5m")

        llm = ChatOllama(
            model=model_name,
            temperature=0,
            base_url=ollama_base_url,
            num_ctx=num_ctx,
            num_predict=num_predict,
            keep_alive=keep_alive
        )

        logger.info(
            "LLM yüklendi: %s, ctx=%s, predict=%s, keep_alive=%s",
            model_name,
            num_ctx,
            num_predict,
            keep_alive
        )

        return llm

    except Exception:
        logger.exception("LLM yüklenirken hata oluştu.")
        return None


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