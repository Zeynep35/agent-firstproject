from fastapi import FastAPI, UploadFile, File, Form, Query, Depends, HTTPException, status
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer, OAuth2PasswordRequestForm
import os

from datetime import datetime, timedelta, timezone
import jwt

from dotenv import load_dotenv
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

load_dotenv()

app = FastAPI(
    title="AgentDemo V3 API",
    version="3.1.0"
)

llm = get_llm()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-jwt-secret")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "120"))

DEMO_USERNAME = os.getenv("DEMO_USERNAME", "zeynep")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "123456")

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="login"
)

API_KEY = os.getenv("AGENTDEMO_API_KEY", "dev-secret-key")

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)


def verify_api_key(api_key: str = Depends(api_key_header)):
    """
    Basit API Key authentication.
    /health hariç endpointlerde kullanılacak.
    """

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key eksik. X-API-Key header göndermelisin."
        )

    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Geçersiz API key."
        )

    return api_key


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"

class RagRequest(BaseModel):
    question: str
    user_id: str = "default_user"

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

def create_access_token(data: dict):
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=JWT_EXPIRE_MINUTES
    )

    payload = data.copy()
    payload.update(
        {
            "exp": expire
        }
    )

    token = jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM
    )

    return token


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM]
        )

        user_id = payload.get("sub")

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token içinde kullanıcı bilgisi yok."
            )

        return {
            "user_id": user_id
        }

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token süresi dolmuş."
        )

    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Geçersiz token."
        )

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "AgentDemo V3 API"
    }

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Kullanıcı login endpointi.
    Swagger'daki Authorize butonu bu endpoint üzerinden token alabilir.
    """

    if (
        form_data.username != DEMO_USERNAME
        or form_data.password != DEMO_PASSWORD
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Kullanıcı adı veya şifre hatalı."
        )

    access_token = create_access_token(
        {
            "sub": form_data.username
        }
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": form_data.username
    }

@app.post("/chat", dependencies=[Depends(verify_api_key)])
def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):

    user_id = current_user["user_id"]

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

@app.get("/pdfs", dependencies=[Depends(verify_api_key)])
def get_pdfs(
    include_public: bool = Query(True),
     current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]

    try:
        vectorstore = load_existing_vectorstore()
        pdfs = list_indexed_pdfs(
            vectorstore,
            user_id=user_id,
            include_public=include_public
        )

        return {
            "pdfs": pdfs,
            "count": len(pdfs),
            "user_id": user_id,
            "include_public": include_public
        }

    except Exception as e:
        logger.exception("PDF listesi alınırken hata oluştu.")
        return {
            "pdfs": [],
            "count": 0,
            "user_id": user_id,
            "include_public": include_public,
            "error": str(e)
        }

@app.post("/rag-chat", dependencies=[Depends(verify_api_key)])
def rag_chat(request: RagRequest,current_user: dict = Depends(get_current_user)):
    
    user_id = current_user["user_id"]

    try:
        vectorstore = load_existing_vectorstore()

        answer, sources = ask_rag(
            question=request.question,
            vectorstore=vectorstore,
            llm=llm,
            user_id=request.user_id
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

@app.post("/chat/stream", dependencies=[Depends(verify_api_key)])
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
                    llm=llm,
                    user_id=request.user_id
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

@app.post("/upload-pdf", dependencies=[Depends(verify_api_key)])
async def upload_pdf(
    file: UploadFile = File(..., description="Yüklenecek PDF dosyası"),
    use_vision: bool = Form(False),
    max_vision_pages: int = Form(1),
    user_id: str = Form("default_user"),
    visibility: str = Form("private"),
    current_user: dict = Depends(get_current_user)
):
    
    user_id = current_user["user_id"]

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
            max_vision_pages=max_vision_pages,
            user_id=user_id,
            visibility=visibility
        )

        pdfs = list_indexed_pdfs(vectorstoreuser_id=user_id, include_public=True)

        return {
            "message": message,
            "pdfs": pdfs,
            "count": len(pdfs),
            "user_id": user_id,
            "visibility": visibility
        }

    except Exception as e:
        logger.exception("API PDF yükleme sırasında hata oluştu.")

        return {
            "message": f"PDF yüklenirken hata oluştu: {e}",
            "pdfs": [],
            "count": 0
        }

@app.delete("/pdfs", dependencies=[Depends(verify_api_key)])
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
    
@app.post("/clear-pdfs", dependencies=[Depends(verify_api_key)])
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