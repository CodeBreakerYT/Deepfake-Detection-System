FROM python:3.10-slim

# Install system dependencies for audio processing and computer vision
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up user 1000 to run on Hugging Face Spaces securely
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /app

# Copy dependency list and install packages under user local
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy source code
COPY --chown=user . .

# Expose default HF Spaces port
EXPOSE 7860

# Run flask server (will pick up PORT=7860 env var from HF Space automatically)
CMD ["python", "app.py"]
