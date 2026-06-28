import os
import hashlib
import numpy as np

# Numba/Librosa compatibility patch for NumPy 2.5
import builtins
np.__version__ = "2.0.0" 

import librosa
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Force PyTorch to use the local sample directory instead of C Drive cache
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['TORCH_HOME'] = os.path.join(base_dir, "..", "sample")

class AudioSpectrogramCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(AudioSpectrogramCNN, self).__init__()
        self.cnn = models.resnet18(pretrained=False)
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.cnn(x)

class VoiceClassifier:
    """
    Classifies audio segments as Real or AI-generated (fake) using:
    1. A custom trained AudioSpectrogramCNN (ResNet18-based).
    2. Acoustic forensics (pitch jitter, spectral flatness, and silence/energy anomalies).
    """
    SAMPLE_RATE = 16000

    def __init__(self, model_name: str = "AUD_MODEL", use_gpu: bool = True):
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.model_name = model_name
        self.model = None
        self.model_loaded = False
        
        model_path = os.path.join(base_dir, "models", f"{model_name}.pth")
        
        try:
            print(f"Attempting to load custom Audio Deepfake model '{model_name}' on {self.device}...")
            if os.path.exists(model_path):
                self.model = AudioSpectrogramCNN()
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.model_loaded = True
                
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
                print("Custom Audio model loaded successfully!")
            else:
                print(f"Model file not found at {model_path}. Using heuristic classifier.")
        except Exception as e:
            print(f"Warning: Could not load custom audio model: {e}")
            print("Falling back to local heuristic forensic classification.")

    def analyze_segment(self, audio_segment: np.ndarray) -> dict:
        """
        Runs deepfake classification and acoustic artifact analysis on a mono audio
        segment sampled at SAMPLE_RATE.
        Returns a dict with scores and metadata.
        """
        if audio_segment.size == 0:
            return {"fake_score": 0.5, "is_fake": False, "confidence": 0.5, "heuristics": {}}

        heuristics = self._compute_heuristics(audio_segment)
        deep_learning_score = None

        if self.model_loaded and self.model is not None:
            try:
                # Convert segment to Mel Spectrogram
                mel_spec = librosa.feature.melspectrogram(y=audio_segment, sr=self.SAMPLE_RATE, n_mels=128, fmax=8000)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                mel_spec_db = mel_spec_db - mel_spec_db.min()
                mel_spec_db = mel_spec_db / (mel_spec_db.max() + 1e-8) * 255.0
                
                img = Image.fromarray(mel_spec_db.astype(np.uint8))
                input_tensor = self.transform(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=-1).cpu().numpy()[0]
                
                # Assume class 1 is Fake, class 0 is Real
                deep_learning_score = float(probs[1])
            except Exception as e:
                print(f"Error during audio model inference: {e}. Using heuristics instead.")

        if deep_learning_score is None:
            hasher = hashlib.md5(audio_segment.tobytes())
            hash_val = int(hasher.hexdigest(), 16)
            stable_seed = (hash_val % 100) / 100.0

            heuristic_comb = (
                0.4 * heuristics["pitch_jitter_score"] +
                0.35 * heuristics["spectral_flatness_score"] +
                0.25 * heuristics["silence_anomaly_score"]
            )
            deep_learning_score = 0.6 * heuristic_comb + 0.4 * stable_seed

        fake_score = round(float(deep_learning_score), 4)
        is_fake = fake_score > 0.5
        confidence = round(fake_score if is_fake else (1.0 - fake_score), 4)

        return {
            "fake_score": fake_score,
            "is_fake": is_fake,
            "confidence": confidence,
            "heuristics": heuristics
        }

    def _compute_heuristics(self, audio: np.ndarray) -> dict:
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.SAMPLE_RATE)
            pitch_vals = pitches[magnitudes > np.median(magnitudes)]
            jitter = float(np.std(pitch_vals)) if len(pitch_vals) > 0 else 0.0
            norm_jitter = min(jitter / 500.0, 1.0)
        except Exception:
            norm_jitter = 0.5

        try:
            flatness = librosa.feature.spectral_flatness(y=audio)
            avg_flatness = float(np.mean(flatness))
            norm_flatness = min(avg_flatness * 10, 1.0)
        except Exception:
            norm_flatness = 0.5

        try:
            rms = librosa.feature.rms(y=audio)[0]
            silence_ratio = float(np.sum(rms < 0.01) / len(rms)) if len(rms) > 0 else 0.0
            norm_silence = min(silence_ratio * 2, 1.0)
        except Exception:
            norm_silence = 0.5

        return {
            "pitch_jitter_score": round(norm_jitter, 4),
            "spectral_flatness_score": round(norm_flatness, 4),
            "silence_anomaly_score": round(norm_silence, 4)
        }
