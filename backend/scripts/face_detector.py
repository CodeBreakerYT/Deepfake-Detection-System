import cv2
import numpy as np

class FaceDetector:
    """
    Detects faces in images/frames using OpenCV's Haar Cascade classifier.
    """
    def __init__(self):
        # Path to Haar cascade file included in opencv-python
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade for face detection.")

    def detect_faces(self, frame_rgb: np.ndarray) -> list[dict]:
        """
        Detects faces in an RGB image.
        Returns a list of dicts: {'box': [x, y, w, h], 'face_crop': np.ndarray}
        """
        # OpenCV Haar cascade expects grayscale and BGR layout for speed
        # But we pass RGB, so convert to grayscale directly
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        detected = []
        for (x, y, w, h) in faces:
            # Add small padding to crop to capture full facial features
            pad_h = int(h * 0.1)
            pad_w = int(w * 0.1)
            
            y_start = max(0, y - pad_h)
            y_end = min(frame_rgb.shape[0], y + h + pad_h)
            x_start = max(0, x - pad_w)
            x_end = min(frame_rgb.shape[1], x + w + pad_w)
            
            face_crop = frame_rgb[y_start:y_end, x_start:x_end]
            
            detected.append({
                "box": [int(x), int(y), int(w), int(h)],
                "padded_box": [int(x_start), int(y_start), int(x_end - x_start), int(y_end - y_start)],
                "face_crop": face_crop
            })
            
        return detected
