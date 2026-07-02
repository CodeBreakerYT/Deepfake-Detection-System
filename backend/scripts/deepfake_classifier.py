import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

# Keep HF model downloads inside the repo instead of polluting the user's C: drive cache
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(base_dir, "..", "hf_cache"))

from transformers import AutoImageProcessor, AutoModelForImageClassification

# Face-forgery specialist: ViT fine-tuned on real vs deepfake face datasets.
FACE_MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
# General AI-image generalist: Swin fine-tuned to catch diffusion-generated imagery
# (Midjourney/SDXL/etc.), not limited to faces. Complements the face specialist above.
GENERAL_MODEL_ID = "Organika/sdxl-detector"


class DeepfakeClassifier:
    """
    Classifies images/frames/face-crops as Real or AI-generated using an ensemble of
    two pretrained transformer models:
    1. A face-forgery specialist (ViT) for classic deepfake/face-swap detection.
    2. A general AI-generated-image detector (Swin) for diffusion-model content
       (Midjourney, Stable Diffusion, DALL-E, etc.) that isn't necessarily a face.
    Falls back to pixel-level forensic heuristics if the models fail to load.
    """
    def __init__(self, use_gpu: bool = True):
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

        self.face_model = None
        self.face_processor = None
        self.face_fake_idx = 1

        self.general_model = None
        self.general_processor = None
        self.general_fake_idx = 0

        try:
            print(f"Loading face-forgery model '{FACE_MODEL_ID}' on {self.device}...")
            self.face_processor = AutoImageProcessor.from_pretrained(FACE_MODEL_ID)
            self.face_model = AutoModelForImageClassification.from_pretrained(FACE_MODEL_ID)
            self.face_model.to(self.device).eval()
            self.face_fake_idx = self._find_fake_index(self.face_model.config.id2label)
            print("Face-forgery model loaded successfully!")
        except Exception as e:
            print(f"Warning: could not load face-forgery model: {e}")
            self.face_model = None

        try:
            print(f"Loading general AI-image model '{GENERAL_MODEL_ID}' on {self.device}...")
            self.general_processor = AutoImageProcessor.from_pretrained(GENERAL_MODEL_ID)
            self.general_model = AutoModelForImageClassification.from_pretrained(GENERAL_MODEL_ID)
            self.general_model.to(self.device).eval()
            self.general_fake_idx = self._find_fake_index(self.general_model.config.id2label)
            print("General AI-image model loaded successfully!")
        except Exception as e:
            print(f"Warning: could not load general AI-image model: {e}")
            self.general_model = None

        self.model_loaded = self.face_model is not None or self.general_model is not None
        if not self.model_loaded:
            print("No deep learning models available. Falling back to forensic heuristics only.")

    @staticmethod
    def _find_fake_index(id2label: dict) -> int:
        """Finds the class index representing 'fake/AI-generated' from a model's id2label map."""
        for idx, label in id2label.items():
            l = str(label).lower()
            if any(k in l for k in ("fake", "artificial", "synthetic", "generated", "ai")):
                return int(idx)
        return 1

    def _run_model(self, model, processor, fake_idx: int, image_rgb: np.ndarray) -> float:
        pil_img = Image.fromarray(image_rgb).convert("RGB")
        inputs = processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        return float(probs[fake_idx])

    def analyze_video_sequence(self, frames_list) -> float:
        """
        Runs the general AI-image model over a subsample of raw RGB video frames and
        returns the mean fake probability. Used as a whole-frame signal to complement
        the per-face analysis (catches AI-generated video content with no detectable face).
        """
        if self.general_model is None or not frames_list:
            return None

        try:
            seq_length = 12
            total = len(frames_list)
            step = max(1, total // seq_length)
            indices = list(range(0, total, step))[:seq_length]

            scores = []
            for i in indices:
                try:
                    scores.append(self._run_model(self.general_model, self.general_processor, self.general_fake_idx, frames_list[i]))
                except Exception as e:
                    print(f"Error scoring frame {i} in video sequence: {e}")

            if not scores:
                return None
            return float(np.mean(scores))
        except Exception as e:
            print(f"Error during video sequence inference: {e}")
            return None

    def analyze_face(self, face_rgb: np.ndarray, is_face: bool = True) -> dict:
        """
        Runs deepfake classification and calculates pixel-level artifacts.
        `is_face` should be True only when face_rgb is an actual detected face crop;
        pass False for whole-frame images (e.g. no face found), since the face-forgery
        specialist is unreliable outside its face-crop training domain.
        Returns a dict with scores and metadata.
        """
        if face_rgb.size == 0:
            return {"fake_score": 0.5, "is_fake": False, "confidence": 0.5, "heuristics": {}}

        # 1. Compute pixel-level heuristics (forensics)
        heuristics = self._compute_heuristics(face_rgb)

        # 2. Run the pretrained model ensemble
        face_score = None
        general_score = None
        if is_face and self.face_model is not None:
            try:
                face_score = self._run_model(self.face_model, self.face_processor, self.face_fake_idx, face_rgb)
            except Exception as e:
                print(f"Error during face-forgery model inference: {e}")

        if self.general_model is not None:
            try:
                general_score = self._run_model(self.general_model, self.general_processor, self.general_fake_idx, face_rgb)
            except Exception as e:
                print(f"Error during general AI-image model inference: {e}")

        if face_score is not None and general_score is not None:
            # Face specialist is weighted more heavily on actual face crops; the general
            # detector is the far more reliable signal outside of that domain.
            deep_learning_score = 0.6 * face_score + 0.4 * general_score
        elif general_score is not None:
            deep_learning_score = general_score
        elif face_score is not None:
            deep_learning_score = face_score
        else:
            # No deep model available: fall back to pure forensic heuristics (no randomness).
            deep_learning_score = (
                0.4 * heuristics["blur_artifact_score"] +
                0.4 * heuristics["frequency_anomaly_score"] +
                0.2 * heuristics["color_anomaly_score"]
            )

        # Final score aggregation
        fake_score = round(float(deep_learning_score), 4)
        is_fake = fake_score > 0.5
        confidence = round(fake_score if is_fake else (1.0 - fake_score), 4)

        return {
            "fake_score": fake_score,
            "is_fake": is_fake,
            "confidence": confidence,
            "heuristics": heuristics,
            "used_vit_model": self.model_loaded
        }

    def _compute_heuristics(self, face_rgb: np.ndarray) -> dict:
        """
        Computes pixel-level forensic metrics:
        - Blurriness (Laplacian Variance): Deepfakes often have blurred/blended boundaries.
        - High Frequency Energy (FFT): Deepfakes display spectral anomalies/smoothing.
        - Color Anomaly (Skin tone variance/histogram outliers).
        """
        # Convert to grayscale
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)

        # A. Laplacian Blur detection
        # Variance of Laplacian represents sharpness. Low variance = blurry/synthetic blending
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(laplacian.var())
        # Map lap_var to an artifact score between 0 and 1 (lower sharpness -> higher score)
        # Standard faces are sharp (var > 150), values < 80 are blurry
        blur_artifact = max(0.0, min(1.0, 1.0 - (lap_var / 200.0)))

        # B. Spectral (FFT) analysis
        # Compute 2D Fourier Transform
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-5)

        # Analyze center (low frequency) vs outer boundaries (high frequency)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        # Define mask for low frequency (center area)
        r = min(h, w) // 10
        r = max(5, r)

        low_freq = magnitude_spectrum[cy-r:cy+r, cx-r:cx+r]
        high_freq_mask = np.ones_like(magnitude_spectrum)
        high_freq_mask[cy-r:cy+r, cx-r:cx+r] = 0
        high_freq = magnitude_spectrum * high_freq_mask

        mean_low = float(np.mean(low_freq))
        mean_high = float(np.sum(high_freq) / (np.sum(high_freq_mask) + 1e-5))

        # Ratio of low frequency to high frequency.
        # Deepfakes often lack high frequencies, so this ratio is abnormally high.
        freq_ratio = mean_low / (mean_high + 1e-5)
        # Normal faces ratio is around 1.2 - 2.2. Ratio > 2.8 indicates missing details/smoothing.
        freq_anomaly = max(0.0, min(1.0, (freq_ratio - 1.5) / 1.5))

        # C. Color distribution & Lighting anomalies
        # Natural skin has high red channel mean and consistent color variances.
        # Deepfakes often exhibit green/blue channel mismatches or extreme uniformity.
        r_chan = face_rgb[:, :, 0]
        g_chan = face_rgb[:, :, 1]
        b_chan = face_rgb[:, :, 2]

        r_std = float(r_chan.std())
        g_std = float(g_chan.std())
        b_std = float(b_chan.std())

        # Compare channel variances. Unnatural faces have overly similar channel standard deviations
        # (flat color tones) or extreme differences.
        std_diff = abs(r_std - g_std) + abs(g_std - b_std)
        # If std_diff is very small (< 2) or standard deviations are very low, color is too uniform.
        mean_std = (r_std + g_std + b_std) / 3.0

        color_anomaly = 0.0
        if mean_std < 15.0:  # flat texture
            color_anomaly += 0.5
        if std_diff < 1.5:  # no skin depth / lighting variations
            color_anomaly += 0.5
        color_anomaly = min(1.0, color_anomaly)

        return {
            "sharpness_val": round(lap_var, 2),
            "blur_artifact_score": round(blur_artifact, 4),
            "freq_ratio": round(freq_ratio, 4),
            "frequency_anomaly_score": round(freq_anomaly, 4),
            "color_depth_val": round(std_diff, 2),
            "color_anomaly_score": round(color_anomaly, 4)
        }
