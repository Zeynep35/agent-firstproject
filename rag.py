import io
import os
import tempfile
from uuid import uuid4
import hashlib
from vision import describe_image
from logger_config import logger

import fitz
import pytesseract
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


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

def load_pdf_with_vision(
        pdf_path, 
        file_name, 
        file_hash=None, 
        file_size=None, 
        max_pages=1,
        user_id="default_user",
        visibility="private"
    ):
    """
    PDF sayfalarını görsele çevirir,
    vision model ile açıklar ve Document listesi döndürür.
    """

    vision_docs = []
    pdf = None

    try:
        pdf = fitz.open(pdf_path)

        total_pages = min(len(pdf), max_pages)

        for page_index in range(total_pages):
            page = pdf[page_index]

            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_image:
                tmp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                image_path = tmp_image.name
                tmp_image.close()

                pix.save(image_path)

            try:
                vision_text = describe_image(
                    image_path=image_path,
                    question=(
                        "Bu PDF sayfasını görsel olarak analiz et. "
                        "Sayfadaki karakterleri, görselleri, renkleri, şekilleri, tabloları "
                        "ve okunabilir yazıları Türkçe açıkla."
                        
                    )
                )

                if vision_text and vision_text.strip():
                    vision_docs.append(
                        Document(
                            page_content=(
                                "Vision görsel açıklaması:\n"
                                + vision_text.strip()
                            ),
                            metadata={
                                "source": file_name,
                                "file_name": file_name,
                                "page": page_index + 1,
                                "file_hash": file_hash,
                                "file_size": file_size,
                                "extraction_type": "vision",
                                "user_id": user_id,
                                "visibility": visibility

                            }
                        )
                    )

            except Exception:
                logger.exception("Vision analizi sırasında hata oluştu.")

            finally:
                if os.path.exists(image_path):
                    os.remove(image_path)

    except Exception:
        logger.exception("PDF vision için görsele çevrilirken hata oluştu.")

    finally:
        if pdf is not None:
            pdf.close()

    return vision_docs


def create_vectorstore_from_pdfs(
    uploaded_files,
    use_vision=False,
    max_vision_pages=3,
    enable_vision=None,
    vision_max_pages=None,
    user_id="default_user",
    visibility="private"
):
    """
    Streamlit'ten gelen çoklu PDF dosyalarını okur,
    duplicate PDF'leri engeller,
    sayfalara böler, chunklara ayırır ve Chroma'ya ekler.
    Normal PDF metni yoksa OCR ile okumayı dener.
    """

    if enable_vision is not None:
        use_vision = enable_vision

    if vision_max_pages is not None:
        max_vision_pages = vision_max_pages

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

            if use_vision:
                vision_docs = load_pdf_with_vision(
                    pdf_path=tmp_path,
                    file_name=uploaded_file.name,
                    file_hash=file_hash,
                    max_pages=max_vision_pages,
                    user_id=user_id,
                    visibility=visibility
                )

                docs.extend(vision_docs)

            for doc in docs:
                doc.metadata["source"] = file_name
                doc.metadata["file_name"] = file_name
                doc.metadata["file_hash"] = file_hash
                doc.metadata["file_size"] = file_size
                doc.metadata["user_id"] = user_id
                doc.metadata["visibility"] = visibility

                page_value = doc.metadata.get("page", 0)

                try:
                    page_value = int(page_value)
                except Exception:
                    page_value = 0

                if doc.metadata.get("extraction_type") in ["ocr", "vision"]:
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

    vision_page_count = sum(
        1 for doc in all_docs
        if doc.metadata.get("extraction_type") == "vision"
    )

    message = f"{len(processed_files)} PDF işlendi. {len(chunks)} parça Chroma'ya eklendi."

    if skipped_files:
        message += " Atlanan PDF'ler: " + ", ".join(skipped_files)

    if ocr_page_count > 0:
        message += f" OCR ile okunan sayfa sayısı: {ocr_page_count}."
    
    if vision_page_count > 0:
        message += f" Vision ile yorumlanan sayfa sayısı: {vision_page_count}."

    return vectorstore, message

def filter_docs_for_user(docs, user_id="default_user", include_public=True):
    """
    RAG sonuçlarını kullanıcıya göre filtreler.

    Kullanıcı:
    - kendi private PDF'lerini görebilir
    - public PDF'leri görebilir
    - başkasının private PDF'lerini göremez
    """

    filtered_docs = []

    for doc in docs:
        metadata = doc.metadata or {}

        doc_user_id = metadata.get("user_id")
        visibility = metadata.get("visibility", "private")

        # Eski metadata'sız kayıtlar geçici uyumluluk için görünür.
        if doc_user_id is None:
            filtered_docs.append(doc)
            continue

        # Kullanıcının kendi belgesi
        if doc_user_id == user_id:
            filtered_docs.append(doc)
            continue

        # Public belge
        if include_public and visibility == "public":
            filtered_docs.append(doc)
            continue

    return filtered_docs

def ask_rag(question, vectorstore, llm, user_id="default_user"):
    if vectorstore is None:
        return "Önce PDF yüklemen gerekiyor.", []

    if llm is None:
        return "LLM yüklenemedi. agent_core.py içindeki get_llm() fonksiyonunu kontrol et.", []

    docs = vectorstore.similarity_search(
        question,
        k=20
    )

    docs = [
        doc for doc in docs
        if (
            doc.metadata.get("user_id") == user_id
            or doc.metadata.get("visibility") == "public"
            or doc.metadata.get("user_id") is None
        )
    ]

    docs = docs[:8]

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
- Vision görsel açıklaması varsa bunu PDF sayfasının görsel yorumu olarak kullan.
- Kullanıcı görsel soruyorsa ve bağlamda "Vision görsel açıklaması" varsa mutlaka bu açıklamaya göre cevap ver.
- Vision açıklaması kısa veya eksik olsa bile "bulunamadı" deme; mevcut açıklamayı yorumla.
- Cevabın sonunda kullandığın PDF adı, sayfa numarası ve içerik türünü yaz.

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

def stream_rag_answer(question, vectorstore, llm, user_id="default_user"):
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

    docs = vectorstore.similarity_search(
        question,
        k=20
    )

    docs = [
        doc for doc in docs
        if (
            doc.metadata.get("user_id") == user_id
            or doc.metadata.get("visibility") == "public"
            or doc.metadata.get("user_id") is None
        )
    ]

    docs = docs[:8]

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
- Vision görsel açıklaması varsa bunu PDF sayfasının görsel yorumu olarak kullan.
- Kullanıcı görsel soruyorsa ve bağlamda "Vision görsel açıklaması" varsa mutlaka bu açıklamaya göre cevap ver.
- Vision açıklaması kısa veya eksik olsa bile "bulunamadı" deme; mevcut açıklamayı yorumla.
- Cevabın sonunda kullandığın PDF adı, sayfa numarası ve içerik türünü yaz.

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


def list_indexed_pdfs(vectorstore, user_id=None, include_public=True):
    """
    ChromaDB içinde kayıtlı PDF isimlerini listeler.
    user_id verilirse sadece o kullanıcının private PDF'leri ve public PDF'ler döner.
    """

    if vectorstore is None:
        return []

    try:
        data = vectorstore.get(include=["metadatas"])
        metadatas = data.get("metadatas", []) or []

        pdf_names = set()

        for metadata in metadatas:
            if not metadata:
                continue

            file_name = metadata.get("source") or metadata.get("file_name")
            metadata_user_id = metadata.get("user_id")
            visibility = metadata.get("visibility", "private")

            if not file_name:
                continue

             # Streamlit eski kullanım: user_id verilmezse her şeyi göster.
            if user_id is None:
                pdf_names.add(file_name)
                continue

            # Kullanıcının kendi PDF'i  
            if metadata_user_id == user_id:
                pdf_names.add(file_name)
                continue
            
            # Public PDF
            if include_public and visibility == "public":
                pdf_names.add(file_name)
                continue

            # Eski metadata'sız kayıtlar geçici olarak görünsün
            if metadata_user_id is None:
                pdf_names.add(file_name)

        return sorted(pdf_names)

    except Exception:
        logger.exception("PDF listelenirken hata oluştu.")
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