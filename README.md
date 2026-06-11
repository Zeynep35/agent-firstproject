# AgentDemo - Ollama + LangChain + LangGraph + Streamlit + RAG

Bu proje, local LLM kullanarak çalışan basit ama geliştirilebilir bir yapay zeka agent uygulamasıdır. Projede Ollama üzerinden local model çalıştırılır, Streamlit ile arayüz sağlanır, LangGraph ile kısa süreli hafıza eklenir ve PDF dosyaları üzerinden RAG sistemi kullanılır.

## Demo

### PDF RAG Sistemi

![RAG Demo](screenshots/rag-demo.png)

## Özellikler

* Local LLM desteği
* Ollama entegrasyonu
* Streamlit chat arayüzü
* LangChain agent yapısı
* LangGraph short-term memory
* Router tabanlı yönlendirme
* PDF tabanlı RAG sistemi
* ChromaDB vector database
* Ollama embedding desteği
* Basit tool sistemi:

  * Hava durumu
  * Çıkarma işlemi
  * Demo web search

## Kullanılan Teknolojiler

* Python
* Streamlit
* LangChain
* LangGraph
* Ollama
* ChromaDB
* PyPDFLoader
* RecursiveCharacterTextSplitter
* OllamaEmbeddings
* Mistral local model
* nomic-embed-text embedding modeli

## Proje Mimarisi

Proje temel olarak şu akışla çalışır:

```text
Kullanıcı mesajı
      ↓
Streamlit Chat Arayüzü
      ↓
Router Kontrolü
      ↓
 ┌───────────────┬───────────────┬───────────────┐
 │ Hava durumu   │ Basit işlem   │ PDF/RAG       │
 └───────────────┴───────────────┴───────────────┘
      ↓
Gerekirse Agent / LLM
      ↓
Cevap
```

## RAG Mimarisi

PDF dosyası yüklendiğinde sistem şu işlemleri yapar:

```text
PDF yüklenir
    ↓
PyPDFLoader ile okunur
    ↓
Metin parçalara bölünür
    ↓
OllamaEmbeddings ile vektöre çevrilir
    ↓
ChromaDB içine kaydedilir
    ↓
Kullanıcı soru sorar
    ↓
En alakalı belge parçaları bulunur
    ↓
LLM bu parçalara göre cevap üretir
```

## Kurulum

Önce gerekli paketleri kur:

```bash
pip install streamlit langchain langgraph langchain-ollama langchain-community langchain-text-splitters langchain-chroma pypdf
```

Ollama tarafında kullanılacak modelleri indir:

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

## Çalıştırma

Projeyi çalıştırmak için:

```bash
streamlit run app.py
```

## Örnek Kullanım

Normal sohbet:

```text
Benim adım Zeynep.
```

Sonra:

```text
Benim adım ne?
```

Agent, short-term memory sayesinde önceki konuşmayı hatırlayabilir.

PDF/RAG örneği:

```text
PDF yükle
```

Sonra:

```text
Bu belgede ne anlatılıyor?
```

veya:

```text
Bu PDF'e göre ana konu nedir?
```

Router, bu soruyu RAG sistemine yönlendirir.

## Router Mantığı

Projede modelin her şeye kendisinin karar vermesi yerine basit bir router kullanılmıştır. Bunun sebebi local modellerde tool calling davranışının her zaman stabil olmamasıdır.

Router örnek olarak şunları kontrol eder:

```text
hava → hava durumu cevabı
eksi / çıkar → matematik yönlendirmesi
pdf / belge / doküman / dosya → RAG sistemi
diğer mesajlar → normal agent cevabı
```

Bu yapı sayesinde sistem daha hızlı ve daha kontrollü çalışır.

## Short-Term Memory

Projede LangGraph `InMemorySaver` kullanılmıştır.

```python
checkpointer=InMemorySaver()
```

Bu sayede aynı `thread_id` içinde konuşma geçmişi tutulur.

Örnek:

```python
config = {
    "configurable": {
        "thread_id": "zeynep_1"
    }
}
```

Not: Bu hafıza kalıcı değildir. Uygulama kapatıldığında hafıza sıfırlanır. Kalıcı hafıza için ileride SQLite, Postgres, ChromaDB veya FAISS tabanlı long-term memory eklenebilir.

## Mevcut Sınırlamalar

* Local model kullanıldığı için cevap süresi uzun olabilir.
* Mistral bazı Türkçe cevaplarda doğal olmayan çıktılar verebilir.
* Tool calling tamamen modele bırakılmamıştır.
* RAG sadece metin tabanlı PDF'lerde iyi çalışır.
* Görsel/taranmış PDF'ler için OCR desteği yoktur.
* Memory şu an sadece short-term memory olarak çalışır.

## Geliştirilecek Özellikler

* Long-term memory
* SQLite veya Postgres checkpointer
* Daha gelişmiş RAG sistemi
* PDF kaynak sayfa gösterme
* Çoklu PDF desteği
* Daha iyi router yapısı
* Gerçek web search entegrasyonu
* FAISS desteği
* Kullanıcı bazlı thread sistemi
* Chat geçmişini dışa aktarma
* Daha hızlı model seçeneği

## Proje Dosya Yapısı Önerisi

Şu an proje tek dosyada geliştirilebilir. Ancak proje büyüdüğünde aşağıdaki yapıya geçilebilir:

```text
agent-demo/
│
├── app.py
├── agent.py
├── rag.py
├── router.py
├── tools.py
├── requirements.txt
├── README.md
│
├── uploads/
│   └── uploaded.pdf
│
└── chroma_db/
```

## requirements.txt Örneği

```txt
streamlit
langchain
langgraph
langchain-ollama
langchain-community
langchain-text-splitters
langchain-chroma
pypdf
chromadb
```

## Amaç

Bu proje, local LLM kullanarak çalışan bir agent sisteminin temel yapılarını öğrenmek için geliştirilmiştir. Amaç; LangChain, LangGraph, Ollama, memory, router ve RAG kavramlarını tek bir pratik uygulamada birleştirmektir.

