from agentic_rag import should_use_rag, answer_with_agentic_rag
from logger_config import logger
from web_search import answer_with_web_search


def should_use_web_search(user_input: str):
    """
    Kullanıcı güncel/web bilgisi istiyorsa True döner.
    PDF sorularını web'e göndermemek için önce PDF kelimelerini kontrol eder.
    """

    text = user_input.lower()

    pdf_keywords = [
        "pdf",
        "belge",
        "doküman",
        "dosya",
        "yüklediğim",
        "sana verdiğim",
        "bu belgede",
        "bu pdf",
        "bu dosyada"
    ]

    # Kullanıcı PDF/belge diyorsa web'e değil RAG'e gitsin
    if any(keyword in text for keyword in pdf_keywords):
        return False

    web_keywords = [
        "internette ara",
        "internetten ara",
        "webde ara",
        "web'de ara",
        "web ara",
        "google'da ara",
        "online ara",
        "arama yap",
        "araştır",
        "güncel",
        "son dakika",
        "bugün",
        "şu an",
        "haber",
        "fiyat",
        "kaç oldu",
        "2026"
    ]

    return any(keyword in text for keyword in web_keywords)


def router(user_input: str, vectorstore=None, llm=None):
    text = user_input.lower()

    # 1. Gerçek web search
    if should_use_web_search(user_input):
        logger.info("Router: Gerçek web search seçildi.")

        if llm is None:
            return "Web araması için LLM yüklenemedi."

        return answer_with_web_search(
            question=user_input,
            llm=llm
        )

    # 2. PDF / Agentic RAG
    if vectorstore is not None and should_use_rag(user_input):
        logger.info("Router: Agentic RAG seçildi.")

        if llm is None:
            return "PDF cevabı için LLM yüklenemedi."

        answer, sources = answer_with_agentic_rag(
            question=user_input,
            vectorstore=vectorstore,
            llm=llm
        )

        if sources:
            source_text = "\n".join(
                [
                    f"- {source.get('source')} / Sayfa {source.get('page')}"
                    for source in sources
                ]
            )

            return f"{answer}\n\nKaynaklar:\n{source_text}"

        return answer

    # 3. Basit hava demo
    if "hava" in text:
        logger.info("Router: Hava durumu seçildi.")

        if "izmir" in text:
            return "İzmir'de hava güneşli."
        elif "istanbul" in text:
            return "İstanbul için hava bulutlu."
        else:
            return "Hangi şehir için hava durumunu istiyorsun?"

    # 4. Basit çıkarma demo
    if "eksi" in text or "çıkar" in text:
        logger.info("Router: Çıkarma işlemi seçildi.")
        return "Çıkarma işlemi için örnek: 10'dan 3 çıkar."

    # 5. Normal agent akışına bırak
    logger.info("Router: Agent seçildi.")
    return None