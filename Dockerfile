# Base Image
FROM python:3.10-slim


ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1


WORKDIR /code

# System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face + model tools
RUN pip install --no-cache-dir huggingface-hub sentencepiece accelerate fasttext

# Hugging Face cache environment
ENV HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    HUGGINGFACE_HUB_CACHE=/models/huggingface \
    HF_HUB_CACHE=/models/huggingface

# Created cache dir and set permissions
RUN mkdir -p /models/huggingface && chmod -R 777 /models/huggingface

# Pre-download models at build time
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-4B-Instruct-2507')" \
 && python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')" \
 && python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='facebook/fasttext-language-identification', filename='model.bin')" \
 && python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='drrobot9/nllb-ig-yo-ha-finetuned')" \
 && find /models/huggingface -name '*.lock' -delete

# Preload tokenizers (avoid runtime delays)
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507', use_fast=True)" \
 && python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', use_fast=True)" \
 && python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('drrobot9/nllb-ig-yo-ha-finetuned', use_fast=True)"

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 7860

# Run FastAPI app with uvicorn (1 workers for concurrency)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]