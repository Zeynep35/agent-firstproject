from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama



@tool
def get_weather(city: str) -> str:
    """Şehir adına göre hava durumunu döndürür."""
    city = city.lower()

    if city == "izmir":
        return "İzmir'de hava güneşli."
    else:
        return f"{city} için hava bulutlu."

@tool
def web_search(sorgu: str) ->str:
    """İnternette arama yapar."""

    sonuc = "Bulunan sonuçlar..."

    return sonuc 
    

@tool
def eksi_hesapla(sayi1: int, sayi2:int) -> int:
    """Verilen sayılardan ikinci sayıyı birinci sayıdan çıkartır."""

    return sayi1 - sayi2

llm = ChatOllama(
    model="mistral:latest"
)

agent = create_agent(
    model=llm,
    tools=[get_weather, eksi_hesapla, web_search],
    system_prompt="Sen kullanıcıya yardımcı olan, nazikçe istekleri çözüp cevap üreten bir yapay zeka ajanısın."
)



result = agent.invoke(
    {"messages": [{"role": "user", "content": "İzmir'de hava nasıl?"}]}
)

print(result["messages"][-1].content)


