import os
import numpy as np
import librosa
import torch
import torch.nn.functional as F

# Keep HF model downloads inside the repo instead of polluting the user's C: drive cache
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(base_dir, "..", "hf_cache"))

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Wav2Vec2-based voice deepfake/spoof detector, trained on real vs AI-cloned
# speech (ElevenLabs, Amazon Polly, Kokoro, Hume AI, Speechify, Luvvoice, etc.)
AUDIO_MODEL_ID = "garystafford/wav2vec2-deepfake-voice-detector"


class VoiceClassifier:
    """
    Classifies audio segments as Real or AI-generated (fake) using:
    1. A pretrained Wav2Vec2-based voice deepfake/spoof detector.
    2. Acoustic forensics (pitch jitter, spectral flatness, and silence/energy anomalies)
       used as a fallback if the model fails to load.
    """
    SAMPLE_RATE = 16000

    def __init__(self, use_gpu: bool = True):
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.model = None
        self.feature_extractor = None
        self.fake_idx = 1
        self.model_loaded = False

        try:
            print(f"Loading voice deepfake model '{AUDIO_MODEL_ID}' on {self.device}...")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_ID)
            self.model = AutoModelForAudioClassification.from_pretrained(AUDIO_MODEL_ID)
            self.model.to(self.device).eval()
            self.fake_idx = self._find_fake_index(self.model.config.id2label)
            self.model_loaded = True
            print("Voice deepfake model loaded successfully!")
        except Exception as e:
            print(f"Warning: could not load voice deepfake model: {e}")
            print("Falling back to local heuristic acoustic forensic classification.")

    @staticmethod
    def _find_fake_index(id2label: dict) -> int:
        for idx, label in id2label.items():
            l = str(label).lower()
            if any(k in l for k in ("fake", "synthetic", "spoof", "generated", "artificial")):
                return int(idx)
        return 1

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
                inputs = self.feature_extractor(
                    audio_segment, sampling_rate=self.SAMPLE_RATE, return_tensors="pt", padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

                deep_learning_score = float(probs[self.fake_idx])
            except Exception as e:
                print(f"Error during voice deepfake model inference: {e}. Using heuristics instead.")

        if deep_learning_score is None:
            # No deep model available: fall back to pure acoustic heuristics (no randomness).
            deep_learning_score = (
                0.4 * heuristics["pitch_jitter_score"] +
                0.35 * heuristics["spectral_flatness_score"] +
                0.25 * heuristics["silence_anomaly_score"]
            )

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
