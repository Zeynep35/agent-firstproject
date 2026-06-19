from logger_config import logger


def should_use_rag(user_input: str) -> bool:
    """
    Kullanıcının sorusu PDF / belge araması gerektiriyor mu?
    Basit karar fonksiyonu.
    """
    text = user_input.lower()

    rag_keywords = [
        "pdf",
        "belge",
        "doküman",
        "dosya",
        "yüklediğim",
        "sana verdiğim",
        "bu metinde",
        "bu belgede",
        "bu dokümanda",
        "bu pdf",
        "metne göre",
        "belgeye göre",
        "dosyaya göre",
        "özetle",
        "özet çıkar",
        "ne anlatıyor",
        "ne hakkında",
    ]

    return any(keyword in text for keyword in rag_keywords)


def retrieve_docs(question: str, vectorstore, k: int = 5):
    """
    Chroma içinden soruya en yakın PDF parçalarını getirir.
    """
    if vectorstore is None:
        return []

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    docs = retriever.invoke(question)

    return docs


def format_docs_for_prompt(docs):
    """
    Gelen doküman parçalarını LLM promptuna uygun metne çevirir.
    """
    if not docs:
        return ""

    formatted_docs = []

    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Bilinmeyen kaynak")
        page = doc.metadata.get("page", "Bilinmeyen sayfa")
        content = doc.page_content

        formatted_docs.append(
            f"""
[Parça {index}]
Kaynak: {source}
Sayfa: {page}
İçerik:
{content}
"""
        )

    return "\n\n".join(formatted_docs)


def extract_sources(docs):
    """
    Kaynak listesini tekrar etmeyecek şekilde çıkarır.
    """
    sources = []

    for doc in docs:
        item = {
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page")
        }

        if item not in sources:
            sources.append(item)

    return sources


def rewrite_query(user_input: str, llm):
    """
    İlk arama zayıfsa, kullanıcı sorusunu PDF araması için daha iyi hale getirir.
    """
    prompt = f"""
Aşağıdaki kullanıcı sorusunu PDF içinde arama yapmak için daha net ve kısa bir arama sorgusuna çevir.

Kurallar:
- Sadece arama sorgusunu yaz.
- Açıklama yapma.
- Türkçe yaz.

Kullanıcı sorusu:
{user_input}
"""

    response = llm.invoke(prompt)
    rewritten = response.content if hasattr(response, "content") else str(response)

    return rewritten.strip()


def answer_with_agentic_rag(user_input: str, vectorstore, llm):
    """
    Agentic RAG akışı:
    1. İlk arama yapar.
    2. Sonuç zayıfsa sorguyu yeniden yazar.
    3. Tekrar arar.
    4. Bulduğu PDF parçalarıyla cevap üretir.
    """

    if vectorstore is None:
        return "Önce PDF yüklemen gerekiyor.", []

    if llm is None:
        return "LLM bulunamadı. app.py içinde get_llm() ve router bağlantısını kontrol et.", []

    logger.info("Agentic RAG: İlk arama başlatıldı.")

    docs = retrieve_docs(
        question=user_input,
        vectorstore=vectorstore,
        k=5
    )

    # İlk aramada hiç sonuç yoksa sorguyu yeniden yazıp tekrar dene
    if not docs:
        logger.info("Agentic RAG: İlk aramada sonuç yok. Sorgu yeniden yazılıyor.")

        new_query = rewrite_query(user_input, llm)

        logger.info(f"Agentic RAG: Yeni sorgu: {new_query}")

        docs = retrieve_docs(
            question=new_query,
            vectorstore=vectorstore,
            k=5
        )

    if not docs:
        return "Bu bilgi PDF'lerde bulunamadı.", []

    context = format_docs_for_prompt(docs)

    if not context.strip():
        return "PDF bulundu ama içinden okunabilir metin çıkarılamadı. Bu PDF taranmış/görsel PDF olabilir.", []

    final_prompt = f"""
Sen PDF belgelerini analiz eden bir asistansın.

Aşağıdaki PDF parçalarına göre cevap ver.

Kurallar:
- Sadece verilen PDF içeriğine dayan.
- PDF içeriğinde olmayan bilgiyi uydurma.
- Cevabı kısa, net ve Türkçe ver.
- Eğer bilgi açıkça yoksa "Bu bilgi PDF'lerde bulunamadı." de.
- Cevabın sonunda kaynak belirt.

PDF İçeriği:
{context}

Kullanıcı Sorusu:
{user_input}
"""

    logger.info("Agentic RAG: Final cevap üretiliyor.")

    response = llm.invoke(final_prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    sources = extract_sources(docs)

    return answer, sources