FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-tur \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads chroma_db screenshots

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]