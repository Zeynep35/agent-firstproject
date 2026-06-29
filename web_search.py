import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch


load_dotenv()


def web_search(query: str, max_results: int = 5):
    """
    Gerçek internet araması yapar.
    Tavily API kullanır.
    """

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return "TAVILY_API_KEY bulunamadı. .env dosyasını kontrol et."

    search_tool = TavilySearch(
        max_results=max_results,
        topic="general"
    )

    results = search_tool.invoke({
        "query": query
    })

    return results


def format_web_results(results):
    """
    Web sonuçlarını LLM'e verilecek temiz metne çevirir.
    """

    if isinstance(results, str):
        return results

    if not results:
        return "Web aramasında sonuç bulunamadı."

    formatted_results = []

    # Tavily bazen dict içinde results döndürür
    items = results.get("results", results) if isinstance(results, dict) else results

    for index, item in enumerate(items, start=1):
        title = item.get("title", "Başlık yok")
        url = item.get("url", "URL yok")
        content = item.get("content", "")

        formatted_results.append(
            f"""
Sonuç {index}
Başlık: {title}
URL: {url}
İçerik: {content}
"""
        )

    return "\n".join(formatted_results)

def answer_with_web_search(question: str, llm):
    """
    Kullanıcının sorusunu web'de arar ve LLM'e kaynaklı cevap ürettirir.
    """

    results = web_search(question)
    web_context = format_web_results(results)

    prompt = f"""
Sen gerçek web araması yapabilen bir asistansın.

Kurallar:
- Sadece aşağıdaki web sonuçlarına göre cevap ver.
- Güncel bilgi istendiğinde web sonuçlarını esas al.
- Bilgi web sonuçlarında yoksa "Web sonuçlarında bu bilgi bulunamadı." de.
- Cevabın sonunda kullandığın kaynak URL'leri listele.

Web Sonuçları:
{web_context}

Kullanıcı Sorusu:
{question}

Cevap:
"""

    response = llm.invoke(prompt)

    answer = response.content if hasattr(response, "content") else str(response)

    return answer