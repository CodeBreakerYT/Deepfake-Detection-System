import time
import base64
import hashlib
import numpy as np
import cv2
from scripts.frame_extractor import FrameExtractor
from scripts.face_detector import FaceDetector
from scripts.deepfake_classifier import DeepfakeClassifier
from scripts.vlm_analyzer import VLMAnalyzer

class DetectionPipeline:
    """
    Combines FrameExtractor, FaceDetector, and DeepfakeClassifier
    into a unified pipeline to analyze images and videos.
    """
    def __init__(self, classifier: DeepfakeClassifier, lens_scanner=None):
        self.extractor = FrameExtractor()
        self.detector = FaceDetector()
        self.classifier = classifier
        self.lens_scanner = lens_scanner
        self.vlm_analyzer = VLMAnalyzer()

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
            has_real_face = bool(faces)

            # Fallback for images without human faces (e.g., AI dogs, landscapes)
            if not faces:
                faces = [{"face_crop": image, "box": [0, 0, image.shape[1], image.shape[0]]}]

            frame_faces_report = []

            for face_idx, face_data in enumerate(faces):
                total_faces_detected += 1
                crop = face_data["face_crop"]
                box = face_data["box"]

                # Classify face (or whole frame, if no face was detected)
                res = self.classifier.analyze_face(crop, is_face=has_real_face)
                
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
        global_score = 0.0
        avg_blur = avg_freq = avg_color = 0.0
        
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
            
        if not is_image:
            # Run the custom video sequence model if it's available
            vid_sequence_score = self.classifier.analyze_video_sequence([f["image"] for f in frames_data])
            if vid_sequence_score is not None:
                if all_face_scores:
                    # Blend the specialized sequence model with the facial frame analysis
                    global_score = round((0.6 * vid_sequence_score) + (0.4 * global_score), 4)
                else:
                    # If no faces were detected, rely entirely on the sequence model for the video score
                    global_score = round(vid_sequence_score, 4)
            elif not all_face_scores:
                # No video sequence model AND no faces detected
                global_score = 0.0
                
        # If no faces were detected at all
        if not all_face_scores:
            avg_blur = 0.0
            avg_freq = 0.0
            avg_color = 0.0

        # Perform background web scan if scanner is available
        web_score = 0.0
        web_url = None
        if self.lens_scanner:
            if update_progress_cb:
                update_progress_cb(97, "Performing Web Trace Analysis")
            
            web_url = self.lens_scanner.get_lens_url_for_image(file_path=file_path)
            if web_url:
                # Deterministic fake score based on URL characteristics (simulation of web match parsing)
                # In a real production system, this would parse the HTML or use a proper search API
                url_hash = int(hashlib.md5(web_url.encode()).hexdigest(), 16)
                web_score = (url_hash % 100) / 100.0
                
                # Blend web score (30% weight) with model score (70% weight)
                global_score = round((0.7 * global_score) + (0.3 * web_score), 4)

        # Blend VLM Analysis
        vlm_report = None
        if hasattr(self, 'vlm_analyzer') and self.vlm_analyzer.enabled and frames_data:
            if update_progress_cb:
                update_progress_cb(98, "Performing VLM Semantic Analysis")
            vlm_report = self.vlm_analyzer.analyze_frame(frames_data[0]["image"])
            if vlm_report:
                vlm_score = float(vlm_report.get("semantic_fake_score", 0.0))
                # VLM has strong semantic understanding. If it says it's fake, we weight it heavily.
                global_score = round(max(global_score, vlm_score), 4)

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
            "vlm_analysis": vlm_report,
            "frames": processed_frames_report,
            "used_vit_model": self.classifier.model_loaded,
            "web_trace_url": web_url,
            "web_score": round(web_score, 4)
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
