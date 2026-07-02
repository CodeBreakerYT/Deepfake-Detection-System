import os
import uuid
import shutil
import hashlib
import threading
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from dotenv import load_dotenv

# Load env variables from .env file
load_dotenv()

from scripts.deepfake_classifier import DeepfakeClassifier
from scripts.pipeline import DetectionPipeline
from scripts.voice_classifier import VoiceClassifier
from scripts.voice_pipeline import VoiceDetectionPipeline
from scripts.firebase_manager import FirebaseManager
from scripts.google_lens_scanner import GoogleLensScanner

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)

# Enable CORS
allowed_origins = os.environ.get("FRONTEND_ORIGINS", "*").split(",")
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

# Initialize components
print("Initializing pipeline services...")
db_manager = FirebaseManager()
classifier = DeepfakeClassifier()
lens_scanner = GoogleLensScanner()
pipeline = DetectionPipeline(classifier, lens_scanner=lens_scanner)
voice_classifier = VoiceClassifier()
voice_pipeline = VoiceDetectionPipeline(voice_classifier)
print("All services initialized!")

def get_file_hash_from_stream(stream) -> str:
    """
    Computes SHA-256 hash of upload file without loading entire file in memory.
    """
    sha256 = hashlib.sha256()
    while chunk := stream.read(4096):
        sha256.update(chunk)
    stream.seek(0)
    return sha256.hexdigest()

def async_analyze_video(task_id: str, temp_path: str, file_hash: str, filename: str):
    """
    Background worker function for video analysis.
    """
    try:
        def update_progress(progress_pct: int, stage_name: str):
            db_manager.update_task_status(file_hash, {
                "status": "processing",
                "progress": progress_pct,
                "stage": stage_name,
                "task_id": task_id,
                "filename": filename
            })

        # Run analysis
        report = pipeline.analyze_media(temp_path, is_image=False, update_progress_cb=update_progress)
        
        category = "fake" if report.get("is_fake", False) else "real"
        
        doc_data = {
            "name": filename,
            "category": category,
            "result": report,
            "status": "completed",
            "progress": 100,
            "stage": "Finished",
            "task_id": task_id
        }
        
        # Cache completed result
        db_manager.cache_result(file_hash, doc_data)
        
        # We don't need to update task status separately because cache_result uses the same document
    except Exception as e:
        print(f"Error in background video processing task {task_id}: {e}")
        db_manager.update_task_status(file_hash, {
            "status": "failed",
            "progress": 0,
            "error": str(e),
            "task_id": task_id
        })
    finally:
        # Clean up video file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def async_analyze_audio(task_id: str, temp_path: str, file_hash: str, filename: str):
    """
    Background worker function for audio/voice analysis.
    """
    try:
        def update_progress(progress_pct: int, stage_name: str):
            db_manager.update_task_status(file_hash, {
                "status": "processing",
                "progress": progress_pct,
                "stage": stage_name,
                "task_id": task_id,
                "filename": filename
            })

        report = voice_pipeline.analyze_audio(temp_path, update_progress_cb=update_progress)

        category = "fake" if report.get("is_fake", False) else "real"
        
        doc_data = {
            "name": filename,
            "category": category,
            "result": report,
            "status": "completed",
            "progress": 100,
            "stage": "Finished",
            "task_id": task_id
        }

        db_manager.cache_result(file_hash, doc_data)
    except Exception as e:
        print(f"Error in background audio processing task {task_id}: {e}")
        db_manager.update_task_status(file_hash, {
            "status": "failed",
            "progress": 0,
            "error": str(e),
            "task_id": task_id
        })
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route("/", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "Deepfake Detection API (Flask)",
        "database": "firestore" if db_manager.db else "in-memory"
    })

@app.route("/api/detect/audio", methods=["POST"])
def detect_voice_deepfake():
    """
    Uploads an audio file for voice deepfake detection. Checks cache first;
    always processed asynchronously via background thread.
    """
    if "file" not in request.files:
        return jsonify({"detail": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"detail": "No file selected"}), 400

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    is_audio = ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"]

    if not is_audio:
        return jsonify({"detail": "Unsupported file format. Please upload an audio file (wav, mp3, flac, ogg, m4a, aac)."}), 400

    try:
        file_hash = get_file_hash_from_stream(file.stream)
    except Exception as e:
        return jsonify({"detail": f"Failed to process file: {str(e)}"}), 500

    cached_result = db_manager.get_cached_result(file_hash)
    if cached_result:
        print(f"Cache hit for file hash: {file_hash}")
        cached_result["timestamp"] = time_tracker_helper()
        db_manager.cache_result(file_hash, cached_result)
        cached_result["source"] = "cache"
        return jsonify(cached_result)

    task_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{task_id}{ext}")
    file.save(temp_path)

    db_manager.update_task_status(file_hash, {
        "status": "processing",
        "progress": 10,
        "stage": "Initializing background task",
        "task_id": task_id,
        "filename": filename
    })

    # Start async analysis in a background thread
    threading.Thread(
        target=async_analyze_audio,
        args=(task_id, temp_path, file_hash, filename),
        daemon=True
    ).start()

    return jsonify({
        "status": "processing",
        "task_id": task_id,
        "message": "Voice analysis queued in background."
    })

@app.route("/api/detect", methods=["POST"])
def detect_deepfake():
    """
    Uploads an image or video file for deepfake detection.
    Checks cache first; executes synchronously for images and asynchronously for videos.
    """
    if "file" not in request.files:
        return jsonify({"detail": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"detail": "No file selected"}), 400

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    is_image = ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    is_video = ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]

    if not is_image and not is_video:
        return jsonify({"detail": "Unsupported file format. Please upload an image or video."}), 400

    try:
        file_hash = get_file_hash_from_stream(file.stream)
    except Exception as e:
        return jsonify({"detail": f"Failed to process file: {str(e)}"}), 500

    cached_result = db_manager.get_cached_result(file_hash)
    if cached_result:
        print(f"Cache hit for file hash: {file_hash}")
        cached_result["timestamp"] = time_tracker_helper()
        db_manager.cache_result(file_hash, cached_result)
        cached_result["source"] = "cache"
        return jsonify(cached_result)

    task_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{task_id}{ext}")
    file.save(temp_path)

    if is_image:
        # Images are processed synchronously (usually < 1s)
        try:
            report = pipeline.analyze_media(temp_path, is_image=True)
            
            import base64
            from PIL import Image
            import io
            
            try:
                # Resize and base64 encode the image
                with Image.open(temp_path) as img:
                    img.thumbnail((512, 512))
                    buffered = io.BytesIO()
                    img_format = "PNG" if ext == ".png" else "JPEG"
                    img.save(buffered, format=img_format)
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            except Exception as e:
                print(f"Error thumbnailing image: {e}")
                with open(temp_path, "rb") as img_file:
                    img_str = base64.b64encode(img_file.read()).decode("utf-8")
            
            category = "fake" if report.get("is_fake", False) else "real"
            
            doc_data = {
                "image": img_str,
                "name": filename,
                "category": category,
                "result": report,
                "status": "completed"
            }
            db_manager.cache_result(file_hash, doc_data)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            doc_data["source"] = "fresh"
            return jsonify(doc_data)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({"detail": f"Pipeline error: {str(e)}"}), 500
    else:
        # Videos are sent to background threads
        db_manager.update_task_status(file_hash, {
            "status": "processing",
            "progress": 10,
            "stage": "Initializing background task",
            "task_id": task_id,
            "filename": filename
        })
        threading.Thread(
            target=async_analyze_video,
            args=(task_id, temp_path, file_hash, filename),
            daemon=True
        ).start()
        return jsonify({
            "status": "processing",
            "task_id": task_id,
            "message": "Video analysis queued in background."
        })

@app.route("/api/status/<task_id>", methods=["GET"])
def get_status(task_id):
    """
    Polls the progress status of a background video detection job.
    """
    status = db_manager.get_task_status(task_id)
    if not status:
        return jsonify({"detail": "Task not found."}), 404
    return jsonify(status)

@app.route("/api/history", methods=["GET"])
def get_history():
    """
    Fetches the history of completed analyses.
    """
    return jsonify(db_manager.get_history())

@app.route("/api/clear", methods=["POST"])
def clear_cache():
    """
    Clears all local and Firebase cached records.
    """
    db_manager.clear_cache()
    return jsonify({"message": "Cache and task history cleared."})

@app.route("/api/search-web", methods=["POST"])
def search_web_lens():
    """
    Accepts an image file or a base64 string and returns the Google Lens URL.
    """
    data = request.get_json(silent=True)
    base64_str = data.get("image") if data else None

    # If it's a multipart form data with file
    if "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            import tempfile
            import os
            temp_path = os.path.join(tempfile.gettempdir(), f"lens_{uuid.uuid4().hex}.jpg")
            file.save(temp_path)
            url = lens_scanner.get_lens_url_for_image(file_path=temp_path)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if url:
                return jsonify({"lens_url": url})
            return jsonify({"detail": "Failed to generate Google Lens URL"}), 500

    if base64_str:
        url = lens_scanner.get_lens_url_for_image(base64_str=base64_str)
        if url:
            return jsonify({"lens_url": url})
        return jsonify({"detail": "Failed to generate Google Lens URL"}), 500

    return jsonify({"detail": "No file or base64 image provided"}), 400

def time_tracker_helper():
    import time
    return time.time()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
