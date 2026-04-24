# 1. Image de base NVIDIA avec CUDA 11.8 et cuDNN (version "runtime" pour être plus légère)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Évite que l'installation apt-get ne pose des questions interactives
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
# 2. Installation de Python 3.10 et des dépendances système OCR/OpenCV
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    # Dépendances critiques pour OpenCV
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Forcer le système à utiliser la commande 'python' pour appeler 'python3'
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# 3. Installation des frameworks IA (Alignés sur CUDA 11.8)
# On installe d'abord PyTorch et Paddle pour bénéficier du cache Docker
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir paddlepaddle-gpu==2.6.2 -f https://www.paddlepaddle.org.cn/whl/cu118/stable.html

# 4. Copie et installation du reste des dépendances
COPY requirements_backend.txt .
RUN pip install --no-cache-dir -r requirements_backend.txt
RUN python -m spacy download fr_core_news_sm

RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='fr', use_angle_cls=True, use_gpu=False)"

# 5. Copie du code source
COPY . .

# Expose le port de FastAPI
EXPOSE 8000

# Commande pour lancer FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]