import io
import os
import tempfile
from uuid import uuid4
import hashlib

import fitz
import pytesseract
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Chroma veritabanının kaydedileceği klasör
PERSIST_DIR = "./chroma_db"

# Chroma içindeki koleksiyon adı
COLLECTION_NAME = "multi_pdf_rag"


def get_embeddings():
    """
    Embedding modelini oluşturur.
    PDF parçalarını vektöre çevirmek için kullanılır.
    """
    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_BASE_URL
    )

def calculate_file_hash(file_bytes: bytes):
    """
    PDF dosyasının içeriğinden benzersiz hash üretir.
    Aynı dosya tekrar yüklenirse aynı hash çıkar.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def get_indexed_pdf_signatures(vectorstore):
    """
    ChromaDB içindeki mevcut PDF hash, dosya adı ve dosya boyutlarını getirir.
    Eski kayıtlar file_hash içermeyebilir, bu yüzden isim ve boyut da kontrol edilir.
    """
    existing_hashes = set()
    existing_name_size_pairs = set()

    if vectorstore is None:
        return existing_hashes, existing_name_size_pairs

    try:
        data = vectorstore.get(include=["metadatas"])
        metadatas = data.get("metadatas", []) or []

        for metadata in metadatas:
            if not metadata:
                continue

            file_hash = metadata.get("file_hash")
            file_name = metadata.get("file_name") or metadata.get("source")
            file_size = metadata.get("file_size")

            if file_hash:
                existing_hashes.add(file_hash)

            if file_name and file_size:
                existing_name_size_pairs.add((file_name, int(file_size)))

    except Exception:
        pass

    return existing_hashes, existing_name_size_pairs


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

def preprocess_image_for_ocr(image):
    """
    OCR doğruluğunu artırmak için görüntüyü temizler.
    """
    image = image.convert("L")
    image = ImageOps.autocontrast(image)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    image = image.filter(ImageFilter.SHARPEN)

    return image

def load_pdf_with_ocr_fallback(pdf_path, file_name):
    """
    Önce normal PDF metnini okur.
    Eğer PDF tarama/görsel PDF ise OCR ile okumaya çalışır.
    """

    docs = []

    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
    except Exception:
        docs = []

    has_text = any(
        doc.page_content and doc.page_content.strip()
        for doc in docs
    )

    # Normal PDF ise OCR'a gerek yok
    if has_text:
        return docs

    # Metin yoksa OCR dene
    ocr_docs = []
    pdf = None

    try:
        pdf = fitz.open(pdf_path)

        for page_index, page in enumerate(pdf):
            pix = page.get_pixmap(
            matrix=fitz.Matrix(4, 4),
            alpha=False
        )

            image_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(image_bytes))

            image = preprocess_image_for_ocr(image)

            text = pytesseract.image_to_string(
                image,
                lang="tur+eng",
                config="--oem 3 --psm 6"
            )

            print("OCR TEXT:", text[:500])

            if text and text.strip():
                ocr_docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_name,
                            "file_name": file_name,
                            "page": page_index + 1,
                            "extraction_type": "ocr"
                        }
                    )
                )

    finally:
        if pdf is not None:
            pdf.close()

    return ocr_docs


def create_vectorstore_from_pdfs(uploaded_files):
    """
    Streamlit'ten gelen çoklu PDF dosyalarını okur,
    duplicate PDF'leri engeller,
    sayfalara böler, chunklara ayırır ve Chroma'ya ekler.
    Normal PDF metni yoksa OCR ile okumayı dener.
    """

    if not uploaded_files:
        return None, "PDF yüklenmedi."

    vectorstore = load_existing_vectorstore()

    existing_hashes, existing_name_size_pairs = get_indexed_pdf_signatures(vectorstore)

    seen_hashes_this_batch = set()
    seen_name_size_this_batch = set()

    all_docs = []
    skipped_files = []
    processed_files = []

    for uploaded_file in uploaded_files:
        tmp_path = None

        try:
            file_bytes = uploaded_file.getvalue()
            file_hash = calculate_file_hash(file_bytes)
            file_name = uploaded_file.name
            file_size = len(file_bytes)
            name_size_pair = (file_name, file_size)

            # 1. Aynı yükleme içinde aynı içerik tekrar seçilmiş mi?
            if file_hash in seen_hashes_this_batch:
                skipped_files.append(f"{file_name} (aynı yükleme içinde tekrar)")
                continue

            # 2. Aynı yükleme içinde aynı isim + boyut tekrar seçilmiş mi?
            if name_size_pair in seen_name_size_this_batch:
                skipped_files.append(f"{file_name} (aynı isim ve boyutla tekrar seçildi)")
                continue

            # 3. Daha önce ChromaDB'ye aynı içerik eklenmiş mi?
            if file_hash in existing_hashes:
                skipped_files.append(f"{file_name} (zaten eklenmiş)")
                continue

            # 4. Eski kayıtlar file_hash içermeyebilir, isim + boyut fallback kontrolü
            if name_size_pair in existing_name_size_pairs:
                skipped_files.append(f"{file_name} (aynı isim ve boyutla zaten eklenmiş)")
                continue

            seen_hashes_this_batch.add(file_hash)
            seen_name_size_this_batch.add(name_size_pair)
            processed_files.append(file_name)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name

            docs = load_pdf_with_ocr_fallback(
                tmp_path,
                file_name
            )

            for doc in docs:
                doc.metadata["source"] = file_name
                doc.metadata["file_name"] = file_name
                doc.metadata["file_hash"] = file_hash
                doc.metadata["file_size"] = file_size

                page_value = doc.metadata.get("page", 0)

                try:
                    page_value = int(page_value)
                except Exception:
                    page_value = 0

                if doc.metadata.get("extraction_type") == "ocr":
                    doc.metadata["page"] = max(page_value, 1)
                else:
                    doc.metadata["page"] = page_value + 1

            all_docs.extend(docs)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    all_docs = [
        doc for doc in all_docs
        if doc.page_content and doc.page_content.strip()
    ]

    if not all_docs:
        if skipped_files:
            return vectorstore, "Yeni PDF eklenmedi. Atlanan PDF'ler: " + ", ".join(skipped_files)

        return vectorstore, "PDF okundu ama içinden metin çıkarılamadı. OCR da metin bulamadı."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(all_docs)

    if not chunks:
        return vectorstore, "PDF işlendi ama chunk oluşturulamadı."

    ids = [str(uuid4()) for _ in chunks]

    vectorstore.add_documents(
        documents=chunks,
        ids=ids
    )

    ocr_page_count = sum(
        1 for doc in all_docs
        if doc.metadata.get("extraction_type") == "ocr"
    )

    message = f"{len(processed_files)} PDF işlendi. {len(chunks)} parça Chroma'ya eklendi."

    if skipped_files:
        message += " Atlanan PDF'ler: " + ", ".join(skipped_files)

    if ocr_page_count > 0:
        message += f" OCR ile okunan sayfa sayısı: {ocr_page_count}."

    return vectorstore, message

def ask_rag(question, vectorstore, llm):
    if vectorstore is None:
        return "Önce PDF yüklemen gerekiyor.", []

    if llm is None:
        return "LLM yüklenemedi. agent_core.py içindeki get_llm() fonksiyonunu kontrol et.", []

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 8}
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
Sen bir PDF analiz asistanısın.

Kurallar:
- Sadece aşağıdaki PDF bağlamına göre cevap ver.
- Bağlamda açıkça bulunmayan bilgiyi tahmin etme.
- Emin değilsen "Bu bilgi PDF içinde açıkça bulunamadı." de.
- OCR hataları olabilir; bu yüzden cevabını kaynak cümlelere dayandır.
- Cevabın sonunda kullandığın PDF adı ve sayfa numarasını yaz.

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
            "page": doc.metadata.get("page"),
            "extraction_type": doc.metadata.get("extraction_type", "text")
        }
        for doc in docs
    ]

    return answer, sources

def stream_rag_answer(question, vectorstore, llm):
    """
    PDF RAG cevabını token token stream eder.
    Streamlit tarafında gerçek zamanlı yazdırmak için kullanılır.
    """

    if vectorstore is None:
        yield "Önce PDF yüklemen gerekiyor."
        return

    if llm is None:
        yield "LLM yüklenemedi. agent_core.py içindeki get_llm() fonksiyonunu kontrol et."
        return

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 8}
    )

    docs = retriever.invoke(question)

    if not docs:
        yield "Bu bilgi PDF'lerde bulunamadı."
        return

    context = "\n\n".join(
        [
            f"Kaynak: {doc.metadata.get('source')} | Sayfa: {doc.metadata.get('page')}\n{doc.page_content}"
            for doc in docs
        ]
    )

    prompt = f"""
Aşağıdaki PDF içeriklerine göre cevap ver.
Sen bir PDF analiz asistanısın.

Kurallar:
- Sadece aşağıdaki PDF bağlamına göre cevap ver.
- Bağlamda açıkça bulunmayan bilgiyi tahmin etme.
- Emin değilsen "Bu bilgi PDF içinde açıkça bulunamadı." de.
- OCR hataları olabilir; bu yüzden cevabını kaynak cümlelere dayandır.
- Cevabın sonunda kullandığın PDF adı ve sayfa numarasını yaz.

Bağlam:
{context}

Soru:
{question}

Cevap:
"""

    for chunk in llm.stream(prompt):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)

        if token:
            yield token

    sources = [
        {
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "extraction_type": doc.metadata.get("extraction_type", "text")
        }
        for doc in docs
    ]

    if sources:
        yield "\n\n---\nKaynaklar:\n"

        for source in sources:
            yield (
                f"- {source.get('source')} / "
                f"Sayfa {source.get('page')} / "
                f"{source.get('extraction_type')}\n"
            )


def list_indexed_pdfs(vectorstore):
    """
    ChromaDB içinde kayıtlı PDF isimlerini listeler.
    """
    if vectorstore is None:
        return []

    try:
        data = vectorstore.get(include=["metadatas"])
        metadatas = data.get("metadatas", []) or []

        pdf_names = sorted(
            {
                metadata.get("source") or metadata.get("file_name")
                for metadata in metadatas
                if metadata and (metadata.get("source") or metadata.get("file_name"))
            }
        )

        return pdf_names

    except Exception:
        return []


def delete_pdf_from_vectorstore(vectorstore, file_name):
    """
    Seçilen PDF'e ait chunkları ChromaDB'den siler.
    """
    if vectorstore is None:
        return vectorstore, "Önce PDF veritabanı yüklenmeli."

    if not file_name:
        return vectorstore, "Silinecek PDF seçilmedi."

    try:
        data = vectorstore.get(
            where={"source": file_name},
            include=["metadatas"]
        )

        ids = data.get("ids", []) or []

        if not ids:
            data = vectorstore.get(
                where={"file_name": file_name},
                include=["metadatas"]
            )
            ids = data.get("ids", []) or []

        if not ids:
            return vectorstore, f"{file_name} için kayıt bulunamadı."

        vectorstore.delete(ids=ids)

        return vectorstore, f"{file_name} PDF'ine ait {len(ids)} parça silindi."

    except Exception as e:
        return vectorstore, f"PDF silinirken hata oluştu: {e}"


def clear_vectorstore(vectorstore):
    """
    ChromaDB içindeki tüm PDF chunklarını temizler.
    """
    if vectorstore is None:
        return vectorstore, "Temizlenecek PDF verisi yok."

    try:
        data = vectorstore.get(include=["metadatas"])
        ids = data.get("ids", []) or []

        if not ids:
            return vectorstore, "Zaten silinecek veri yok."

        vectorstore.delete(ids=ids)

        return vectorstore, f"Toplam {len(ids)} parça silindi. PDF veritabanı temizlendi."

    except Exception as e:
        return vectorstore, f"Tüm veriler silinirken hata oluştu: {e}"