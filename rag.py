import io
import os
import tempfile
from uuid import uuid4
import hashlib
from vision import describe_image
from logger_config import logger
import time 

import fitz
import pytesseract
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

TESSERACT_CMD = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

load_dotenv()

RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
RAG_SEARCH_K = int(os.getenv("RAG_SEARCH_K", "20"))
RAG_FINAL_K = int(os.getenv("RAG_FINAL_K", "6"))
RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))


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
    embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    return OllamaEmbeddings(
        model=embedding_model,
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
    PDF'i hem normal metin olarak hem de gerekirse OCR ile okur.

    Mantık:
    - Önce PyPDFLoader ile gömülü text okunur.
    - ALWAYS_RUN_OCR=true ise text olsa bile OCR da yapılır.
    - Text hiç yoksa OCR otomatik çalışır.
    """

    all_docs = []
    text_docs = []

    always_run_ocr = os.getenv("ALWAYS_RUN_OCR", "true").lower() == "true"

    # 1. Normal text okuma
    try:
        loader = PyPDFLoader(pdf_path)
        loaded_docs = loader.load()

        for doc in loaded_docs:
            content = (doc.page_content or "").strip()

            if not content:
                continue

            metadata = doc.metadata or {}
            page_number = metadata.get("page", 0)

            # PyPDFLoader genelde page'i 0'dan başlatır.
            if isinstance(page_number, int):
                page_number = page_number + 1

            doc.metadata["source"] = file_name
            doc.metadata["file_name"] = file_name
            doc.metadata["page"] = page_number
            doc.metadata["extraction_type"] = "text"

            text_docs.append(doc)

        if text_docs:
            logger.info(
                "%s PDF metin olarak okundu. Sayfa sayısı: %s",
                file_name,
                len(text_docs)
            )

            all_docs.extend(text_docs)

    except Exception:
        logger.warning(
            "%s PDF metin olarak okunamadı, OCR denenecek.",
            file_name,
            exc_info=True
        )

    # 2. OCR gerekiyor mu?
    should_run_ocr = always_run_ocr or not text_docs

    if not should_run_ocr:
        return all_docs

    # 3. OCR okuma
    try:
        import fitz
        from PIL import Image
        import pytesseract
        from langchain_core.documents import Document

        pdf_document = fitz.open(pdf_path)

        ocr_docs = []

        for page_index in range(len(pdf_document)):
            page = pdf_document[page_index]

            pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), alpha=False)

            image = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            try:
                image = preprocess_image_for_ocr(image)
            except Exception:
                logger.warning(
                    "OCR preprocess başarısız, ham görsel kullanılacak.",
                    exc_info=True
                )

            ocr_text = pytesseract.image_to_string(
                image,
                lang="tur+eng",
                config="--oem 3 --psm 6"
            )

            ocr_text = (ocr_text or "").strip()

            if not ocr_text:
                continue

            ocr_docs.append(
                Document(
                    page_content=ocr_text,
                    metadata={
                        "source": file_name,
                        "file_name": file_name,
                        "page": page_index + 1,
                        "extraction_type": "ocr"
                    }
                )
            )

        pdf_document.close()

        if ocr_docs:
            logger.info(
                "%s PDF OCR ile okundu. OCR sayfa sayısı: %s",
                file_name,
                len(ocr_docs)
            )

            all_docs.extend(ocr_docs)

    except Exception:
        logger.exception("%s OCR yapılırken hata oluştu.", file_name)

    if not all_docs:
        logger.warning("%s PDF içinden text veya OCR içerik çıkarılamadı.", file_name)

    return all_docs

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

            image_path = None

            try:
                tmp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                image_path = tmp_image.name
                tmp_image.close()

                pix.save(image_path)

                vision_text = describe_image(
                    image_path=image_path,
                    question=(
                        "Bu PDF sayfasını görsel olarak analiz et. "
                        "SADECE okunabilir yazı arama. Yazı yoksa bile görseli mutlaka açıkla. "
                        "Eğer sayfa boş değilse karakterleri, nesneleri, renkleri, arka planı, "
                        "sayfa düzenini, ikonları, şekilleri, çizimleri, tablo veya görsel öğeleri anlat. "
                        "Cevabında 'görsel öğe yok' deme; önce gördüğün tüm görsel detayları tarif et. "
                        "Eğer gerçekten tamamen boş bir sayfaysa bunu açıkça söyle. "
                        "Cevabı Türkçe ver."
                    )
                )

                if vision_text and vision_text.strip():
                    vision_docs.append(
                        Document(
                            page_content="Vision görsel açıklaması:\n" + vision_text.strip(),
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
                if image_path and os.path.exists(image_path):
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
                try:
                    vision_docs = load_pdf_with_vision(
                        pdf_path=tmp_path,
                        file_name=file_name,
                        file_hash=file_hash,
                        file_size=file_size,
                        max_pages=max_vision_pages,
                        user_id=user_id,
                        visibility=visibility
                    )

                    if vision_docs:
                        docs.extend(vision_docs)

                except Exception:
                        logger.warning(
                            "Vision modeli çalışmadı. PDF text + OCR ile devam edilecek: %s",
                            file_name,
                            exc_info=True
                        )

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

                doc.metadata["page"] = max(page_value, 1)

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
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP
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

def dedupe_docs(docs):
    """
    Aynı veya çok benzer kaynak parçalarını tekrar tekrar LLM'e göndermeyi engeller.
    """

    unique_docs = []
    seen = set()

    for doc in docs:
        metadata = doc.metadata or {}

        key = (
            metadata.get("source"),
            metadata.get("page"),
            metadata.get("extraction_type"),
            doc.page_content[:120]
        )

        if key in seen:
            continue

        seen.add(key)
        unique_docs.append(doc)

    return unique_docs


def retrieve_docs_for_user(
    vectorstore,
    question,
    user_id="default_user",
    include_public=True
):
    """
    Kullanıcıya göre optimize edilmiş RAG araması yapar.

    Önce Chroma seviyesinde metadata filtresi kullanmayı dener.
    Olmazsa Python tarafında filtrelemeye düşer.
    """

    if vectorstore is None:
        return []

    try:
        chroma_filter = {
            "$or": [
                {"user_id": {"$eq": user_id}},
                {"visibility": {"$eq": "public"}}
            ]
        }

        docs = vectorstore.similarity_search(
            question,
            k=RAG_SEARCH_K,
            filter=chroma_filter
        )

        docs = filter_docs_for_user(
            docs,
            user_id=user_id,
            include_public=include_public
        )

        docs = dedupe_docs(docs)
        docs = docs[:RAG_FINAL_K]

        docs = expand_docs_with_same_page_sources(
            vectorstore=vectorstore,
            docs=docs,
            user_id=user_id
        )

        return docs

    except Exception:
        logger.warning(
            "Chroma metadata filter çalışmadı, Python filtrelemeye düşüldü.",
            exc_info=True
        )

        docs = vectorstore.similarity_search(
            question,
            k=max(RAG_SEARCH_K, 40)
        )

        docs = filter_docs_for_user(
            docs,
            user_id=user_id,
            include_public=include_public
        )
        
        docs = dedupe_docs(docs)
        docs = docs[:RAG_FINAL_K]

        docs = expand_docs_with_same_page_sources(
            vectorstore=vectorstore,
            docs=docs,
            user_id=user_id
        )

        return docs

def build_rag_context(docs):
    """
    RAG için LLM'e gönderilecek context'i kontrollü uzunlukta oluşturur.
    """

    context_parts = []
    total_chars = 0

    for doc in docs:
        metadata = doc.metadata or {}

        source = metadata.get("source") or metadata.get("file_name") or "Bilinmeyen kaynak"
        page = metadata.get("page", "?")
        extraction_type = metadata.get("extraction_type", "text")

        content = doc.page_content.strip()

        if not content:
            continue

        block = (
            f"Kaynak: {source} | Sayfa: {page} | Tür: {extraction_type}\n"
            f"{content}\n"
        )

        if total_chars + len(block) > RAG_MAX_CONTEXT_CHARS:
            remaining = RAG_MAX_CONTEXT_CHARS - total_chars

            if remaining > 500:
                context_parts.append(block[:remaining])

            break

        context_parts.append(block)
        total_chars += len(block)

    return "\n---\n".join(context_parts)

def expand_docs_with_same_page_sources(vectorstore, docs, user_id="default_user"):
    """
    RAG bir sayfadan text/ocr/vision bulduysa,
    aynı PDF + aynı sayfadaki diğer extraction_type kayıtlarını da ekler.

    Amaç:
    - Sadece vision'a bağlı kalmamak
    - Aynı sayfanın OCR ve text bilgisini de context'e katmak
    """

    if vectorstore is None or not docs:
        return docs

    try:
        data = vectorstore.get(
            include=["documents", "metadatas"]
        )

        all_documents = data.get("documents", []) or []
        all_metadatas = data.get("metadatas", []) or []

        wanted_pages = set()

        for doc in docs:
            metadata = doc.metadata or {}

            source = metadata.get("source") or metadata.get("file_name")
            page = metadata.get("page")

            if source and page is not None:
                wanted_pages.add((source, page))

        extra_docs = []

        for content, metadata in zip(all_documents, all_metadatas):
            if not metadata:
                continue

            source = metadata.get("source") or metadata.get("file_name")
            page = metadata.get("page")
            metadata_user_id = metadata.get("user_id")
            visibility = metadata.get("visibility", "private")

            if (source, page) not in wanted_pages:
                continue

            # Kullanıcı güvenlik filtresi
            can_read = (
                metadata_user_id == user_id
                or visibility == "public"
                or metadata_user_id is None
            )

            if not can_read:
                continue

            if not content:
                continue

            from langchain_core.documents import Document

            extra_docs.append(
                Document(
                    page_content=content,
                    metadata=metadata
                )
            )

        combined_docs = docs + extra_docs
        combined_docs = dedupe_docs(combined_docs)

        return combined_docs[:RAG_FINAL_K + 4]

    except Exception:
        logger.warning(
            "Aynı sayfa kaynakları genişletilemedi.",
            exc_info=True
        )
        return docs

def clean_llm_answer(answer: str) -> str:
    """
    Küçük local modeller bazen prompt içindeki 'Soru:', 'Sonuç:' gibi
    bölümleri cevaba kopyalar. Bu fonksiyon cevabı temizler.
    """

    if not answer:
        return ""

    text = str(answer).strip()

    # Model bazen kaynaklar kısmını kendisi üretmeye çalışıyor.
    # Biz kaynakları API tarafında ayrıca verdiğimiz için cevaptan temizliyoruz.
    cut_markers = [
        "\nKaynaklar:",
        "\n---\nKaynaklar:",
        "\nSoru:",
        "\nQuestion:",
    ]

    for marker in cut_markers:
        if marker in text:
            text = text.split(marker)[0].strip()

    # Eğer model "Sonuç:" diye başladıysa sadece sonrasını al.
    if "Sonuç:" in text:
        text = text.split("Sonuç:", 1)[1].strip()

    if "Cevap:" in text:
        text = text.split("Cevap:", 1)[1].strip()

    # Gereksiz tekrarları temizle.
    bad_prefixes = [
        "Kaynakta Tür: vision olan açıklamaya göre",
        "Kaynaklarda Tür: vision olan açıklamayı kullanarak;",
        "Aşağıdaki PDF içeriklerine göre",
    ]

    for prefix in bad_prefixes:
        if text.startswith(prefix):
            text = text.replace(prefix, "", 1).strip(" :;-")

    return text.strip()

def ask_rag(question, vectorstore, llm, user_id="default_user", return_metrics=False):
    total_start = time.perf_counter()

    if vectorstore is None:
        if return_metrics:
            return "Önce PDF yüklemen gerekiyor.", [], {}
        return "Önce PDF yüklemen gerekiyor.", []

    if llm is None:
        if return_metrics:
            return "LLM yüklenemedi. agent_core.py içindeki get_llm() fonksiyonunu kontrol et.", [], {}
        return "LLM yüklenemedi. agent_core.py içindeki get_llm() fonksiyonunu kontrol et.", []

    retrieval_start = time.perf_counter()

    docs = retrieve_docs_for_user(
        vectorstore=vectorstore,
        question=question,
        user_id=user_id,
        include_public=True
    )

    retrieval_ms = int((time.perf_counter() - retrieval_start) * 1000)

    if not docs:
        total_ms = int((time.perf_counter() - total_start) * 1000)

        metrics = {
            "retrieval_ms": retrieval_ms,
            "context_ms": 0,
            "llm_ms": 0,
            "total_ms": total_ms,
            "docs_count": 0,
            "context_chars": 0,
            "source_types": {}
        }

        if return_metrics:
            return "Bu bilgi PDF'lerde bulunamadı.", [], metrics

        return "Bu bilgi PDF'lerde bulunamadı.", []

    context_start = time.perf_counter()
    context = build_rag_context(docs)
    context_ms = int((time.perf_counter() - context_start) * 1000)

    prompt = f"""
Aşağıdaki PDF içeriklerine göre cevap ver.
Sen bir PDF analiz asistanısın.

Kurallar:
- Sadece aşağıdaki PDF bağlamına göre cevap ver.
- Bağlamda açıkça bulunmayan bilgiyi tahmin etme.
- Emin değilsen "Bu bilgi PDF içinde açıkça bulunamadı." de.
- OCR hataları olabilir; bu yüzden cevabını kaynak cümlelere dayandır.
- Vision görsel açıklaması varsa bunu PDF sayfasının görsel yorumu olarak kullan.
- Kullanıcı görsel soruyorsa ve bağlamda "Vision görsel açıklaması" varsa mutlaka bu açıklamaya göre cevap ver.
- Vision açıklaması kısa veya eksik olsa bile "bulunamadı" deme; mevcut açıklamayı yorumla.
- Cevabın sonunda kullandığın PDF adı, sayfa numarası ve içerik türünü yaz.
- Kullanıcının sorusunu cevabında tekrar yazma.
- "Soru:" veya "Sonuç:" başlıklarını üretme.
- Kısa, net ve doğrudan cevap ver.
- En fazla 3 cümle yaz.

Bağlam:
{context}

Soru:
{question}
"""

    llm_start = time.perf_counter()
    response = llm.invoke(prompt)
    llm_ms = int((time.perf_counter() - llm_start) * 1000)

    answer = response.content if hasattr(response, "content") else str(response)

    sources = [
        {
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "extraction_type": doc.metadata.get("extraction_type", "text")
        }
        for doc in docs
    ]

    source_types = {}

    for source in sources:
        extraction_type = source.get("extraction_type", "text")
        source_types[extraction_type] = source_types.get(extraction_type, 0) + 1

    total_ms = int((time.perf_counter() - total_start) * 1000)

    metrics = {
        "retrieval_ms": retrieval_ms,
        "context_ms": context_ms,
        "llm_ms": llm_ms,
        "total_ms": total_ms,
        "docs_count": len(docs),
        "context_chars": len(context),
        "source_types": source_types
    }

    logger.info(
        "RAG metrics | user_id=%s | docs=%s | context_chars=%s | retrieval_ms=%s | context_ms=%s | llm_ms=%s | total_ms=%s",
        user_id,
        len(docs),
        len(context),
        retrieval_ms,
        context_ms,
        llm_ms,
        total_ms
    )

    if return_metrics:
        return answer, sources, metrics

    return answer, sources


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