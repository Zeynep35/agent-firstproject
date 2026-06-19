from rag import ask_rag
from logger_config import logger


def router(user_input: str, vectorstore=None, llm=None):
    text = user_input.lower()

    if vectorstore is not None and (
        "pdf" in text
        or "belge" in text
        or "doküman" in text
        or "dosya" in text
        or "bu metinde" in text
        or "bu belgede" in text
        or "sana verdiğim" in text
        or "yüklediğim" in text
    ):
        logger.info("Router: RAG seçildi.")

        answer, sources = ask_rag(
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

    if "hava" in text:
        logger.info("Router: Hava durumu seçildi.")

        if "izmir" in text:
            return "İzmir'de hava güneşli."
        elif "istanbul" in text:
            return "İstanbul için hava bulutlu."
        else:
            return "Hangi şehir için hava durumunu istiyorsun?"

    if "eksi" in text or "çıkar" in text:
        logger.info("Router: Çıkarma işlemi seçildi.")
        return "Çıkarma işlemi için örnek: 10'dan 3 çıkar."

    if "internet" in text or "web" in text or "ara" in text:
        logger.info("Router: Web search demo seçildi.")
        return "Web arama şu an demo modunda: Bulunan sonuçlar..."

    logger.info("Router: Agent seçildi.")
    return None