from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Şehir adına göre hava durumunu döndürür."""
    city = city.lower()

    if city == "izmir":
        return "İzmir'de hava güneşli."
    else:
        return f"{city} için hava bulutlu."


@tool
def web_search(sorgu: str) -> str:
    """İnternette arama yapar."""
    return "Bulunan sonuçlar..."


@tool
def eksi_hesapla(sayi1: int, sayi2: int) -> int:
    """Verilen sayılardan ikinci sayıyı birinci sayıdan çıkartır."""
    return sayi1 - sayi2