FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip poppler-utils \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Frameworks et PaddleOCR
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir paddlepaddle-gpu==2.6.2 -f https://www.paddlepaddle.org.cn/whl/cu118/stable.html
RUN pip install --no-cache-dir paddleocr

COPY requirements_backend.txt .
RUN pip install --no-cache-dir -r requirements_backend.txt
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='fr', use_angle_cls=True, use_gpu=True)"

#RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# Suppression de la ligne SpaCy si inutile
# RUN python -m spacy download fr_core_news_sm 

COPY . .
# Indiquer à tiktoken d'utiliser ce dossier au lieu du dossier temporaire par défaut
ENV TIKTOKEN_CACHE_DIR=/app/tiktoken_cache

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]