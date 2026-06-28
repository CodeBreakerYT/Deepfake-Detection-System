import cv2
import os
import numpy as np

class FrameExtractor:
    """
    Extracts frames from video files at regular intervals using OpenCV.
    """
    def __init__(self, max_frames: int = 30):
        self.max_frames = max_frames

    def extract_frames(self, video_path: str) -> list[dict]:
        """
        Extracts up to max_frames from the video at even intervals.
        Returns a list of dicts containing the frame index, timestamp, and RGB image array.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames <= 0 or fps <= 0:
            # Fallback for video streams/files without index information
            fps = 25.0
            total_frames = 100

        duration = total_frames / fps
        
        # Determine step size to extract max_frames
        num_to_extract = min(self.max_frames, total_frames)
        if num_to_extract <= 1:
            indices = [0]
        else:
            indices = np.linspace(0, total_frames - 1, num_to_extract, dtype=int).tolist()

        frames = []
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Convert to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Scale down large frames to prevent excessive memory/CPU usage
            height, width = frame_rgb.shape[:2]
            max_dim = 960
            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
                width, height = new_width, new_height

            timestamp = idx / fps
            frames.append({
                "frame_idx": int(idx),
                "timestamp": round(timestamp, 2),
                "image": frame_rgb,
                "width": width,
                "height": height
            })

        cap.release()
        return frames
