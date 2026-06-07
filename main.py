from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver  
import streamlit as st



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

@st.cache_resource
def get_agent():
    llm = ChatOllama(
        model="mistral:latest",
        num_gpu=0,
        temperature=0.1
    )

    agent = create_agent(
        model=llm,
        tools=[],
        system_prompt="""
        Sen konuşma geçmişini dikkatlice kullanan bir asistansın.

        Kurallar:
        - Kullanıcı geçmişte kendisi hakkında bilgi verdiyse onu hatırla.
        - "Benim adım ne?" sorusunda geçmiş mesajları kontrol et.
        - Eğer kullanıcı daha önce adını söylediyse o adı söyle.
        - Asla "bilmiyorum" deme eğer geçmişte bilgi varsa.
        """,
        checkpointer=InMemorySaver(),
    )

    return agent

agent = get_agent()

def router(user_input: str):
    text = user_input.lower()

    if "hava" in text:
        if "izmir" in text:
            return "İzmirde hava güneşli."
        elif "istanbul" in text:
            return "İstanbul için hava bulutlu."
        else:
            return "Hangi şehir için hava durumunu istiyorsun?"

    if "eksi" in text or "çıkar" in text:
        return "Çıkarma işlemi için örnek: 10'dan 3 çıkar."

    if "internet" in text or "web" in text or "ara" in text:
        return "Web arama şu an demo modunda: Bulunan sonuçlar..."

    return None


st.title("AgentDemo")

config = {
    "configurable": {
        "thread_id": "zeynep_1"
    }
}

if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Mesaj yaz...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    quick_answer = router(user_input)

    if quick_answer is not None:
        assistant_answer = quick_answer
    else:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config
        )

        assistant_answer = result["messages"][-1].content

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_answer}
    )

    with st.chat_message("assistant"):
        st.write(assistant_answer)

if st.button("hafızayı göster"):
    state = agent.get_state(config)

    st.write(state.values)
