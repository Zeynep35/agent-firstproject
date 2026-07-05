from fastapi import FastAPI
from pydantic import BaseModel

from agent_core import get_llm
from logger_config import logger


app = FastAPI(
    title="AgentDemo V3 API",
    version="3.1.0"
)

llm = get_llm()


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "AgentDemo V3 API"
    }


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        if llm is None:
            return {
                "answer": "LLM yüklenemedi."
            }

        response = llm.invoke(request.message)
        answer = response.content if hasattr(response, "content") else str(response)

        return {
            "answer": answer
        }

    except Exception as e:
        logger.exception("API chat hatası.")
        return {
            "answer": f"Hata oluştu: {e}"
        }