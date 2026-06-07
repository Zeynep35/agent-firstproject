# Agent First Project

Bu proje LangChain kullanılarak oluşturulmuş basit bir AI Agent örneğidir.

Özellikler:
- Hava durumu sorgulama
- Matematik işlemleri
- Web arama simülasyonu

Teknolojiler:
- Python
- LangChain
- OpenAI
- Ollama
  
Amaç: Agent mimarisini ve tool-calling mantığını öğrenmek. Bu proje başlangıç seviyesinde bir AI Agent örneği olarak geliştirilmektedir. İlerleyen aşamalarda memory (hafıza), multi-agent yapıları, gerçek API entegrasyonları, internet araması yapan araçlar (tools) ve daha gelişmiş agent yetenekleri eklenecektir.

Yeni Özellikler

- LangGraph Short-Term Memory
- Konuşma geçmişini hatırlama
- Streamlit Chat Arayüzü
- Tool Support (Weather, Search, Calculator)
- Ollama Local LLM Integration

Not:
Mevcut sürümde hafıza InMemorySaver ile sağlanmaktadır.
Uygulama yeniden başlatıldığında hafıza sıfırlanır.
Kalıcı hafıza (Long-Term Memory) sonraki sürümlerde eklenecektir.