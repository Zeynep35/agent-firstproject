import streamlit as st

from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from agent_core import get_llm
from logger_config import logger

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