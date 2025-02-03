# Dockerfile
FROM python:3.10-slim-bullseye
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1 \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch
RUN pip3 install --no-cache-dir \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    torchaudio==2.0.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Copy application files
COPY . .

# Create output directory with proper permissions
RUN mkdir -p /app/generated_samples && \
    chmod 777 /app/generated_samples

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir "Pillow==10.3.0" "protobuf<4"

# Set environment variables
ENV PYTHONPATH="/app/janus:${PYTHONPATH}"
ENV PROMPT=""
ENV DEBUG="0"

CMD ["python3", "generate_image.py"]
