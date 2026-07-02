FROM python:3.10-slim

# Install system dependencies for audio processing and computer vision
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up user 1000 to run on Hugging Face Spaces securely
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /app

# Copy dependency list and install packages under user local.
# Resolve everything in a single pass against the CPU-only PyTorch index as
# the primary index (falling back to PyPI for non-torch packages): HF Spaces'
# free/CPU tiers have no GPU, and the default PyPI torch wheel drags in
# several GB of unused NVIDIA CUDA libraries (cudnn, cublas, cufft, ...).
# Resolving torch/torchvision in a separate pip call first doesn't stick --
# the later `-r requirements.txt` pass can still silently upgrade them back
# to the CUDA build to satisfy another package's version constraint.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    --timeout 120 --retries 5

# Copy source code
COPY --chown=user . .

# Pre-download the pretrained detection models at build time so the image is
# self-contained: no re-download (and no anonymous HF Hub rate-limit risk) on
# every cold start / Space restart.
RUN python -c "\
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoFeatureExtractor, AutoModelForAudioClassification; \
AutoImageProcessor.from_pretrained('prithivMLmods/Deep-Fake-Detector-v2-Model'); \
AutoModelForImageClassification.from_pretrained('prithivMLmods/Deep-Fake-Detector-v2-Model'); \
AutoImageProcessor.from_pretrained('Organika/sdxl-detector'); \
AutoModelForImageClassification.from_pretrained('Organika/sdxl-detector'); \
AutoFeatureExtractor.from_pretrained('garystafford/wav2vec2-deepfake-voice-detector'); \
AutoModelForAudioClassification.from_pretrained('garystafford/wav2vec2-deepfake-voice-detector'); \
print('Pretrained models cached.')"

# Expose default HF Spaces port
EXPOSE 7860

# gunicorn in production: one worker (models are loaded once per process and
# this app already fans out per-request work onto background threads), several
# threads to serve concurrent requests, and a long timeout since CPU-only
# inference on video/audio can take tens of seconds.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "180", "app:app"]
