import os
import re
import ollama


VISION_MODEL = os.getenv("VISION_MODEL") or "qwen3-vl:2b"


def is_weak_answer(text: str):
    if not text:
        return True

    cleaned = text.strip().lower()

    weak_patterns = [
        "görselde okunabilir yazı yok.",
        "görselde okunabilir yazı yok",
        "okunabilir yazı yok.",
        "okunabilir yazı yok",
        "yazı yok",
        "metin yok"
    ]

    return cleaned in weak_patterns or len(cleaned) < 35


def extract_clean_answer_from_thinking(thinking_text: str):
    """
    qwen3-vl bazen final cevabı content yerine thinking içine yazıyor.
    Thinking'i kullanıcıya göstermeden içinden sadece final Türkçe cevabı çekiyoruz.
    """

    if not thinking_text:
        return ""

    text = thinking_text.replace("<think>", "").replace("</think>", "").strip()

    # Tırnak içindeki Türkçe final cevapları yakala
    quoted_answers = re.findall(
        r'"([^"]*(?:Bu görselde|Görselde|Resimde)[^"]*)"',
        text,
        flags=re.DOTALL
    )

    if quoted_answers:
        answer = quoted_answers[-1].strip()
        return clean_answer(answer)

    # "So the answer should be:" sonrası gelen Türkçe cevabı yakala
    patterns = [
        r"So the answer should be:\s*['\"]?(.*?)(?:Wait,|Let me|Yes,|$)",
        r"answer should be:\s*['\"]?(.*?)(?:Wait,|Let me|Yes,|$)",
        r"cevap.*?:\s*['\"]?(.*?)(?:Wait,|Let me|Yes,|$)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            possible_answer = match.group(1).strip()
            possible_answer = clean_answer(possible_answer)

            if possible_answer:
                return possible_answer

    # Son çare: "Bu görselde..." diye başlayan kısmı yakala
    match = re.search(
        r"(Bu görselde.*?)(?:\n\n|Wait,|Let me|Yes,|$)",
        text,
        flags=re.DOTALL
    )

    if match:
        return clean_answer(match.group(1).strip())

    return ""


def clean_answer(answer: str):
    if not answer:
        return ""

    answer = answer.replace("\\n", " ")
    answer = answer.replace("\n", " ")
    answer = answer.replace('"', "")
    answer = answer.replace("'", "")
    answer = answer.strip()

    # İngilizce düşünme kalıntılarını kes
    cut_words = [
        "Wait,",
        "Let me",
        "Yes,",
        "The user",
        "I need",
        "So in Turkish"
    ]

    for word in cut_words:
        if word in answer:
            answer = answer.split(word)[0].strip()

    # Çok uzadıysa ilk 4 cümleyi al
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    answer = " ".join(sentences[:4]).strip()

    return answer


def describe_image(image_path, question="Bu görselde ne var? Türkçe açıkla."):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")

    prompt = f"""
Görseli dikkatlice incele.

Türkçe kısa ve net cevap ver:
- Görselde ana olarak ne var?
- Renkleri, karakteri, nesneleri veya arka planı açıkla.
- Görselde okunabilir yazı varsa aynen oku.
- Okunabilir yazı yoksa bunu belirt.

Kullanıcı sorusu:
{question}
"""

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_path]
            }
        ],
        options={
            "temperature": 0,
            "num_ctx": 1024,
            "num_predict": 768
        }
    )

    message = response.get("message") if isinstance(response, dict) else response.message

    content = getattr(message, "content", "") or ""
    thinking = getattr(message, "thinking", "") or ""

    content = content.strip()

    # Eğer content düzgün açıklamaysa direkt ver
    if content and not is_weak_answer(content):
        return content

    # Content zayıfsa thinking içinden final cevabı çek
    clean_from_thinking = extract_clean_answer_from_thinking(thinking)

    if clean_from_thinking:
        if "okunabilir yazı yok" not in clean_from_thinking.lower():
            clean_from_thinking += " Görselde okunabilir yazı yok."

        return clean_from_thinking

    # En son fallback
    if content:
        return content

    return "Model görseli işledi ancak temiz açıklama üretemedi."
