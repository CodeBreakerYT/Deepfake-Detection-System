import time
import hashlib
import numpy as np
import librosa
import speech_recognition as sr
import soundfile as sf
import io
from scripts.voice_classifier import VoiceClassifier

class VoiceDetectionPipeline:
    """
    Loads an audio file, splits it into overlapping segments, classifies each
    segment with VoiceClassifier, and aggregates a global report. Mirrors the
    shape of DetectionPipeline (frames -> segments) so the frontend can reuse
    similar timeline/chart components.
    """
    SAMPLE_RATE = VoiceClassifier.SAMPLE_RATE
    SEGMENT_SEC = 3.0
    HOP_SEC = 2.0
    MAX_SEGMENTS = 40

    def __init__(self, classifier: VoiceClassifier):
        self.classifier = classifier

    def analyze_audio(self, file_path: str, update_progress_cb=None) -> dict:
        start_time = time.time()
        file_hash = self._get_file_hash(file_path)

        if update_progress_cb:
            update_progress_cb(10, "Loading Audio")

        y, sample_rate = librosa.load(file_path, sr=self.SAMPLE_RATE, mono=True)
        duration = float(len(y) / sample_rate) if sample_rate else 0.0

        if update_progress_cb:
            update_progress_cb(25, "Segmenting Audio")

        segments = self._segment_audio(y, sample_rate)
        total_segments = len(segments)

        processed_segments = []
        all_scores = []
        all_jitter = []
        all_flatness = []
        all_silence = []

        for idx, (seg_audio, seg_start) in enumerate(segments):
            if update_progress_cb:
                pct = int(25 + (idx / max(1, total_segments)) * 65)
                update_progress_cb(pct, f"Analyzing Segment {idx + 1}/{total_segments}")

            res = self.classifier.analyze_segment(seg_audio)

            all_scores.append(res["fake_score"])
            all_jitter.append(res["heuristics"].get("pitch_jitter_score", 0.0))
            all_flatness.append(res["heuristics"].get("spectral_flatness_score", 0.0))
            all_silence.append(res["heuristics"].get("silence_anomaly_score", 0.0))

            processed_segments.append({
                "segment_idx": idx,
                "start_time": round(seg_start, 2),
                "end_time": round(seg_start + len(seg_audio) / sample_rate, 2),
                "fake_score": res["fake_score"],
                "is_fake": res["is_fake"],
                "confidence": res["confidence"],
                "heuristics": res["heuristics"]
            })

        if update_progress_cb:
            update_progress_cb(95, "Aggregating Report")

        if all_scores:
            avg_fake_score = float(np.mean(all_scores))
            max_fake_score = float(np.max(all_scores))
            global_score = round((0.7 * max_fake_score) + (0.3 * avg_fake_score), 4)
            avg_jitter = float(np.mean(all_jitter))
            avg_flatness = float(np.mean(all_flatness))
            avg_silence = float(np.mean(all_silence))
        else:
            global_score = 0.0
            avg_jitter = avg_flatness = avg_silence = 0.0

        if update_progress_cb:
            update_progress_cb(97, "Transcribing Audio")

        transcript = ""
        try:
            r = sr.Recognizer()
            wav_io = io.BytesIO()
            sf.write(wav_io, y, sample_rate, format='WAV', subtype='PCM_16')
            wav_io.seek(0)
            with sr.AudioFile(wav_io) as source:
                audio_data = r.record(source)
                transcript = r.recognize_google(audio_data)
        except Exception as e:
            transcript = "[Transcription unavailable or no speech detected]"
            print(f"Transcription error: {e}")

        is_fake = global_score > 0.5
        confidence = global_score if is_fake else (1.0 - global_score)
        processing_time = round(time.time() - start_time, 2)

        report = {
            "file_hash": file_hash,
            "filename": file_path.split("/")[-1].split("\\")[-1],
            "is_audio": True,
            "duration_sec": round(duration, 2),
            "global_fake_score": global_score,
            "is_fake": is_fake,
            "confidence": round(confidence, 4),
            "total_segments_analyzed": total_segments,
            "processing_time_sec": processing_time,
            "timestamp": time.time(),
            "average_heuristics": {
                "pitch_jitter_score": round(avg_jitter, 4),
                "spectral_flatness_score": round(avg_flatness, 4),
                "silence_anomaly_score": round(avg_silence, 4)
            },
            "segments": processed_segments,
            "used_voice_model": self.classifier.model_loaded,
            "transcript": transcript
        }

        if update_progress_cb:
            update_progress_cb(100, "Completed")

        return report

    def _segment_audio(self, y: np.ndarray, sr: int) -> list[tuple[np.ndarray, float]]:
        seg_len = int(self.SEGMENT_SEC * sr)
        hop_len = int(self.HOP_SEC * sr)

        if len(y) <= seg_len:
            return [(y, 0.0)]

        segments = []
        pos = 0
        while pos < len(y) and len(segments) < self.MAX_SEGMENTS:
            chunk = y[pos:pos + seg_len]
            if len(chunk) < int(0.5 * sr):
                break
            segments.append((chunk, pos / sr))
            pos += hop_len

        return segments if segments else [(y, 0.0)]

    def _get_file_hash(self, file_path: str) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256.update(byte_block)
        return sha256.hexdigest()
