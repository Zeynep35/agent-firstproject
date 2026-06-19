import os
import tempfile #yuklenen dosları gecici kayıt eder. Ve onun yerine dosya oluşturur. 
from uuid import uuid4 #pdfler benzersiz id verir.

from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# Chroma veritabanının kaydedileceği klasör
PERSIST_DIR = "./chroma_db"

# Chroma içindeki koleksiyon adı
COLLECTION_NAME = "multi_pdf_rag"


def get_embeddings():
    """
    Embedding modelini oluşturur.
    PDF parçalarını vektöre çevirmek için kullanılır.
    """
    return OllamaEmbeddings(model="nomic-embed-text")


def load_existing_vectorstore():
    """
    Daha önce oluşturulmuş Chroma veritabanını yükler.
    Eğer yoksa aynı klasörde yeni bir tane oluşturur.
    """
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )

    return vectorstore


def create_vectorstore_from_pdfs(uploaded_files):
    """
    Streamlit'ten gelen çoklu PDF dosyalarını okur,
    sayfalara böler, chunklara ayırır ve Chroma'ya ekler.
    """

    if not uploaded_files:
        return None, "PDF yüklenmedi."

    vectorstore = load_existing_vectorstore()

    all_docs = []

    for uploaded_file in uploaded_files:
        tmp_path = None

        try:
            # Streamlit uploaded_file bellekte durduğu için önce geçici PDF dosyasına yazıyoruz.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # PDF'i sayfa sayfa oku
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # Her sayfaya kaynak bilgisi ekle
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
                doc.metadata["file_name"] = uploaded_file.name
                doc.metadata["page"] = doc.metadata.get("page", 0) + 1

            all_docs.extend(docs)

        finally:
            # Geçici dosyayı temizle
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not all_docs:
        return vectorstore, "PDF okundu ama içinden metin çıkarılamadı."

    # PDF sayfalarını küçük parçalara böl
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(all_docs)

    if not chunks:
        return vectorstore, "PDF işlendi ama chunk oluşturulamadı."

    # Her chunk için benzersiz ID üret
    ids = [str(uuid4()) for _ in chunks]

    # Chroma'ya ekle
    vectorstore.add_documents(
        documents=chunks,
        ids=ids
    )

    return vectorstore, f"{len(uploaded_files)} PDF işlendi. {len(chunks)} parça Chroma'ya eklendi."


def ask_rag(question, vectorstore, llm):
    if vectorstore is None:
        return "Önce PDF yüklemen gerekiyor.", []

    if llm is None:
        return "LLM yüklenemedi. agent_core.py içindeki get_llm() fonksiyonunu kontrol et.", []

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    docs = retriever.invoke(question)

    if not docs:
        return "Bu bilgi PDF'lerde bulunamadı.", []

    context = "\n\n".join(
        [
            f"Kaynak: {doc.metadata.get('source')} | Sayfa: {doc.metadata.get('page')}\n{doc.page_content}"
            for doc in docs
        ]
    )

    prompt = f"""
Aşağıdaki PDF içeriklerine göre cevap ver.
Eğer cevap içerikte yoksa "Bu bilgi PDF'lerde bulunamadı." de.
Cevabın sonunda hangi PDF ve sayfalardan yararlandığını belirt.

Bağlam:
{context}

Soru:
{question}
"""

    response = llm.invoke(prompt)

    answer = response.content if hasattr(response, "content") else str(response)

    sources = [
        {
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page")
        }
        for doc in docs
    ]

    return answer, sources