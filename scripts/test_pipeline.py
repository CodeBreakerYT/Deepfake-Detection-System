import os
import cv2
import numpy as np
from scripts.deepfake_classifier import DeepfakeClassifier
from scripts.pipeline import DetectionPipeline

def generate_mock_face_image(path: str):
    """
    Generates a mock face-like image (concentric circles for eyes/face)
    and saves it to test pipeline functionality.
    """
    # Create black canvas
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Draw face boundary (circle)
    cv2.circle(img, (150, 150), 120, (200, 200, 200), -1) # Light gray face
    # Draw left eye
    cv2.circle(img, (100, 120), 15, (50, 50, 50), -1)
    # Draw right eye
    cv2.circle(img, (200, 120), 15, (50, 50, 50), -1)
    # Draw mouth
    cv2.ellipse(img, (150, 180), (40, 20), 0, 0, 180, (50, 50, 50), -1)
    
    # Save image
    cv2.imwrite(path, img)
    print(f"Generated mock face image at: {path}")

def run_test():
    print("=== Deepfake Detection System - Pipeline Test ===")
    test_img_path = "test_target_face.jpg"
    
    # 1. Create a dummy image
    generate_mock_face_image(test_img_path)
    
    # 2. Instantiate services
    print("\n[1/3] Initializing Deepfake Classifier...")
    classifier = DeepfakeClassifier(use_gpu=False) # Disable GPU for test speed/compatibility
    
    print("\n[2/3] Initializing Detection Pipeline...")
    pipeline = DetectionPipeline(classifier)
    
    # 3. Analyze mock image
    print("\n[3/3] Analyzing media...")
    try:
        report = pipeline.analyze_media(test_img_path, is_image=True)
        print("\n=== Analysis Report ===")
        print(f"Filename:               {report['filename']}")
        print(f"File Hash:              {report['file_hash']}")
        print(f"Is Image:               {report['is_image']}")
        print(f"Global Fake Score:      {report['global_fake_score']} ({report['global_fake_score']*100:.1f}%)")
        print(f"Verdict:                {'DEEPFAKE' if report['is_fake'] else 'AUTHENTIC'}")
        print(f"Confidence:             {report['confidence']*100:.1f}%")
        print(f"Faces Detected:         {report['total_faces_detected']}")
        print(f"Used ViT Model:         {report['used_vit_model']}")
        print(f"Processing Time:        {report['processing_time_sec']} seconds")
        
        # Print heuristics info
        print("\nHeuristics Breakdowns:")
        for k, v in report['average_heuristics'].items():
            print(f" - {k}: {v}")
            
        print("\nFrames Analyzed Detail:")
        for frame in report['frames']:
            print(f" - Frame #{frame['frame_idx']} contains {len(frame['faces'])} face(s)")
            for idx, face in enumerate(frame['faces']):
                print(f"    * Face #{idx+1} score: {face['fake_score']} | has crop preview (length: {len(face['crop_b64']) if face['crop_b64'] else 0} chars)")

        print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        raise e
    finally:
        # Cleanup
        if os.path.exists(test_img_path):
            os.remove(test_img_path)
            print(f"Removed temporary test file: {test_img_path}")

if __name__ == "__main__":
    run_test()
