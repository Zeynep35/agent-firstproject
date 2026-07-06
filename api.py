from fastapi import FastAPI, UploadFile, File, Form, Query
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from agent_core import get_llm
from logger_config import logger

from rag import (
    create_vectorstore_from_pdfs,
    load_existing_vectorstore,
    list_indexed_pdfs,
    delete_pdf_from_vectorstore,
    clear_vectorstore,
    ask_rag,
    stream_rag_answer
)


app = FastAPI(
    title="AgentDemo V3 API",
    version="3.1.0"
)

llm = get_llm()


class ChatRequest(BaseModel):
    message: str

class RagRequest(BaseModel):
    question: str

class APIUploadedFile:
    """
    FastAPI UploadFile dosyasını,
    rag.py'nin beklediği Streamlit UploadedFile formatına çevirir.

    rag.py şunları bekliyor:
    - uploaded_file.name
    - uploaded_file.getvalue()
    """

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content

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

def format_sources(sources):
    if not sources:
        return ""

    lines = []

    for source in sources:
        lines.append(
            f"- {source.get('source')} / "
            f"Sayfa {source.get('page')} / "
            f"{source.get('extraction_type')}"
        )

    return "\n".join(lines)

@app.get("/pdfs")
def get_pdfs():
    try:
        vectorstore = load_existing_vectorstore()
        pdfs = list_indexed_pdfs(vectorstore)

        return {
            "pdfs": pdfs,
            "count": len(pdfs)
        }

    except Exception as e:
        logger.exception("PDF listesi alınırken hata oluştu.")
        return {
            "pdfs": [],
            "count": 0,
            "error": str(e)
        }

@app.post("/rag-chat")
def rag_chat(request: RagRequest):
    try:
        vectorstore = load_existing_vectorstore()

        answer, sources = ask_rag(
            question=request.question,
            vectorstore=vectorstore,
            llm=llm
        )

        source_text = format_sources(sources)

        if source_text:
            answer = f"{answer}\n\nKaynaklar:\n{source_text}"

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        logger.exception("RAG API cevabı üretilirken hata oluştu.")
        return {
            "answer": f"RAG cevabı üretilirken hata oluştu: {e}",
            "sources": []
        }

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """
    API üzerinden streaming cevap üretir.

    Akış:
    - Eğer PDF/RAG sorusuysa stream_rag_answer kullanır.
    - Değilse normal LLM streaming yapar.
    """

    def generate():
        try:
            vectorstore = load_existing_vectorstore()

            # PDF / RAG streaming
            if vectorstore is not None:
                # Şimdilik RAG endpoint mantığıyla çalıştırıyoruz.
                # Kullanıcı PDF sorusu sorarsa PDF bağlamından cevap akar.
                for token in stream_rag_answer(
                    question=request.message,
                    vectorstore=vectorstore,
                    llm=llm
                ):
                    yield token

                return

            # Normal LLM streaming
            if llm is None:
                yield "LLM yüklenemedi."
                return

            for chunk in llm.stream(request.message):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)

                if token:
                    yield token

        except Exception as e:
            logger.exception("API streaming sırasında hata oluştu.")
            yield f"Streaming cevap üretilirken hata oluştu: {e}"

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(..., description="Yüklenecek PDF dosyası"),
    use_vision: bool = Form(False),
    max_vision_pages: int = Form(1)
):
    """
    API üzerinden tek PDF yükler.

    Özellikler:
    - Duplicate hash kontrolü rag.py içinde yapılır.
    - OCR destekler.
    - use_vision=true ise Vision analizi de yapar.
    """

    try:
        content = await file.read()

        uploaded_files = [
            APIUploadedFile(
                name=file.filename or "uploaded.pdf",
                content=content
            )
        ]

        vectorstore, message = create_vectorstore_from_pdfs(
            uploaded_files,
            use_vision=use_vision,
            max_vision_pages=max_vision_pages
        )

        pdfs = list_indexed_pdfs(vectorstore)

        return {
            "message": message,
            "pdfs": pdfs,
            "count": len(pdfs)
        }

    except Exception as e:
        logger.exception("API PDF yükleme sırasında hata oluştu.")

        return {
            "message": f"PDF yüklenirken hata oluştu: {e}",
            "pdfs": [],
            "count": 0
        }

@app.delete("/pdfs")
def delete_pdf(file_name: str = Query(..., description="Silinecek PDF dosya adı")):
    """
    API üzerinden seçili PDF'i ChromaDB'den siler.

    Örnek:
    DELETE /pdfs?file_name=sirineocrdenemesi.pdf
    """

    try:
        vectorstore = load_existing_vectorstore()

        vectorstore, message = delete_pdf_from_vectorstore(
            vectorstore,
            file_name
        )

        pdfs = list_indexed_pdfs(vectorstore)

        return {
            "message": message,
            "pdfs": pdfs,
            "count": len(pdfs)
        }

    except Exception as e:
        logger.exception("API PDF silme sırasında hata oluştu.")

        return {
            "message": f"PDF silinirken hata oluştu: {e}",
            "pdfs": [],
            "count": 0
        }
    
@app.post("/clear-pdfs")
def clear_pdfs():
    """
    API üzerinden tüm PDF veritabanını temizler.
    """

    try:
        vectorstore = load_existing_vectorstore()

        vectorstore, message = clear_vectorstore(vectorstore)

        pdfs = list_indexed_pdfs(vectorstore)

        return {
            "message": message,
            "pdfs": pdfs,
            "count": len(pdfs)
        }

    except Exception as e:
        logger.exception("API tüm PDF verilerini temizlerken hata oluştu.")

        return {
            "message": f"Tüm PDF verileri temizlenirken hata oluştu: {e}",
            "pdfs": [],
            "count": 0
        }