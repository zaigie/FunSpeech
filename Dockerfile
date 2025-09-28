FROM python:3.10-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

# Install system packages required for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        sox \
        libsox-dev \
        libsndfile1 \
        build-essential

WORKDIR /app

# Pre-install PyTorch CPU wheels to avoid pulling CUDA runtimes
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.1 \
        torchaudio==2.3.1

# Install Python dependencies
COPY dependencies/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY dependencies/CosyVoice/requirements-cpu.txt /tmp/requirements-cosyvoice.txt
RUN pip install -r /tmp/requirements-cosyvoice.txt

# Clean apt packages and cache
RUN apt remove -y build-essential && apt autoremove -y \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Create runtime directories
RUN mkdir -p /app/temp /app/voices /app/logs \
    && chmod +x start.py

EXPOSE 8000

CMD ["python", "start.py"]
