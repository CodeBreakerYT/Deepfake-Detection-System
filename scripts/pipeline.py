import time
import base64
import hashlib
import numpy as np
import cv2
from scripts.frame_extractor import FrameExtractor
from scripts.face_detector import FaceDetector
from scripts.deepfake_classifier import DeepfakeClassifier

class DetectionPipeline:
    """
    Combines FrameExtractor, FaceDetector, and DeepfakeClassifier
    into a unified pipeline to analyze images and videos.
    """
    def __init__(self, classifier: DeepfakeClassifier):
        self.extractor = FrameExtractor()
        self.detector = FaceDetector()
        self.classifier = classifier

    def analyze_media(self, file_path: str, is_image: bool = False, update_progress_cb=None) -> dict:
        """
        Analyzes an image or video file.
        Returns a dictionary report.
        update_progress_cb is a callback: fn(progress_pct, stage_name)
        """
        start_time = time.time()
        
        # Calculate file hash for caching identification
        file_hash = self._get_file_hash(file_path)

        frames_data = []
        
        if is_image:
            if update_progress_cb:
                update_progress_cb(10, "Loading Image")
            
            img_bgr = cv2.imread(file_path)
            if img_bgr is None:
                raise ValueError("Could not read input image.")
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            frames_data = [{
                "frame_idx": 0,
                "timestamp": 0.0,
                "image": img_rgb,
                "width": img_rgb.shape[1],
                "height": img_rgb.shape[0]
            }]
            
            if update_progress_cb:
                update_progress_cb(40, "Extracting Frames")
        else:
            if update_progress_cb:
                update_progress_cb(10, "Initializing Video Capture")
            # Extract frames
            frames_data = self.extractor.extract_frames(file_path)
            if update_progress_cb:
                update_progress_cb(40, f"Extracted {len(frames_data)} Frames")

        # Process frame by frame
        total_frames = len(frames_data)
        processed_frames_report = []
        
        all_face_scores = []
        all_blur_scores = []
        all_freq_scores = []
        all_color_scores = []
        
        total_faces_detected = 0

        for idx, frame_info in enumerate(frames_data):
            frame_idx = frame_info["frame_idx"]
            timestamp = frame_info["timestamp"]
            image = frame_info["image"]
            
            # Progress calculation: range from 40% to 90%
            if update_progress_cb:
                pct = int(40 + (idx / max(1, total_frames)) * 50)
                update_progress_cb(pct, f"Analyzing Frame {idx + 1}/{total_frames}")

            # Detect faces
            faces = self.detector.detect_faces(image)
            frame_faces_report = []

            for face_idx, face_data in enumerate(faces):
                total_faces_detected += 1
                crop = face_data["face_crop"]
                box = face_data["box"]
                
                # Classify face
                res = self.classifier.analyze_face(crop)
                
                # Convert crop to base64 JPEG for inline browser rendering
                success, buffer = cv2.imencode('.jpg', cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                crop_b64 = ""
                if success:
                    crop_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Append face scores
                all_face_scores.append(res["fake_score"])
                all_blur_scores.append(res["heuristics"]["blur_artifact_score"])
                all_freq_scores.append(res["heuristics"]["frequency_anomaly_score"])
                all_color_scores.append(res["heuristics"]["color_anomaly_score"])

                frame_faces_report.append({
                    "face_id": face_idx,
                    "box": box,
                    "fake_score": res["fake_score"],
                    "is_fake": res["is_fake"],
                    "confidence": res["confidence"],
                    "heuristics": res["heuristics"],
                    "crop_b64": crop_b64
                })

            processed_frames_report.append({
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "faces": frame_faces_report,
                "num_faces": len(frame_faces_report)
            })

        # Aggregation stage
        if update_progress_cb:
            update_progress_cb(95, "Aggregating Report")

        # Compute average metrics
        if all_face_scores:
            # We focus on the worst-offending face per frame to calculate the global score,
            # or the highest face score overall to decide if the video has a deepfake.
            # Max face score is standard for deepfake detection since a single fake face invalidates the video.
            avg_fake_score = float(np.mean(all_face_scores))
            max_fake_score = float(np.max(all_face_scores))
            # Global score will be a blend: 70% max score (conservative) + 30% average score
            global_score = round((0.7 * max_fake_score) + (0.3 * avg_fake_score), 4)
            
            avg_blur = float(np.mean(all_blur_scores))
            avg_freq = float(np.mean(all_freq_scores))
            avg_color = float(np.mean(all_color_scores))
        else:
            # No faces detected. If no faces, we default the deepfake score to 0.0, 
            # and set details indicating no faces were found.
            global_score = 0.0
            avg_blur = 0.0
            avg_freq = 0.0
            avg_color = 0.0

        is_fake = global_score > 0.5
        confidence = global_score if is_fake else (1.0 - global_score)
        
        processing_time = round(time.time() - start_time, 2)
        
        report = {
            "file_hash": file_hash,
            "filename": file_path.split("/")[-1].split("\\")[-1],
            "is_image": is_image,
            "global_fake_score": global_score,
            "is_fake": is_fake,
            "confidence": round(confidence, 4),
            "total_frames_analyzed": total_frames,
            "total_faces_detected": total_faces_detected,
            "processing_time_sec": processing_time,
            "timestamp": time.time(),
            "average_heuristics": {
                "blur_artifact_score": round(avg_blur, 4),
                "frequency_anomaly_score": round(avg_freq, 4),
                "color_anomaly_score": round(avg_color, 4)
            },
            "frames": processed_frames_report,
            "used_vit_model": self.classifier.model_loaded
        }
        
        if update_progress_cb:
            update_progress_cb(100, "Completed")
            
        return report

    def _get_file_hash(self, file_path: str) -> str:
        """
        Calculates SHA-256 hash of the file.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256.update(byte_block)
        return sha256.hexdigest()
