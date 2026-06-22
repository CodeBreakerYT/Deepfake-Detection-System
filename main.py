import os
import uuid
import shutil
import hashlib
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from scripts.deepfake_classifier import DeepfakeClassifier
from scripts.pipeline import DetectionPipeline
from scripts.voice_classifier import VoiceClassifier
from scripts.voice_pipeline import VoiceDetectionPipeline
from scripts.redis_manager import RedisManager

# Setup directories
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Deepfake Detection API", version="1.0.0")

# Enable CORS for the frontend. In production, set FRONTEND_ORIGINS to a comma-separated
# list of allowed origins (e.g. "https://deepshield.example.com"); defaults to "*" for local dev.
allowed_origins = os.environ.get("FRONTEND_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
print("Initializing pipeline services...")
db_manager = RedisManager()
classifier = DeepfakeClassifier()
pipeline = DetectionPipeline(classifier)
voice_classifier = VoiceClassifier()
voice_pipeline = VoiceDetectionPipeline(voice_classifier)
print("All services initialized!")

def get_file_hash_from_upload(upload_file: UploadFile) -> str:
    """
    Computes SHA-256 hash of upload file in chunks without loading entire file in memory.
    """
    sha256 = hashlib.sha256()
    # Read in chunks
    while chunk := upload_file.file.read(4096):
        sha256.update(chunk)
    # Reset file pointer
    upload_file.file.seek(0)
    return sha256.hexdigest()

def async_analyze_video(task_id: str, temp_path: str, file_hash: str, filename: str):
    """
    Background worker function for video analysis.
    """
    try:
        def update_progress(progress_pct: int, stage_name: str):
            db_manager.update_task_status(task_id, {
                "status": "processing",
                "progress": progress_pct,
                "stage": stage_name,
                "task_id": task_id,
                "filename": filename
            })

        # Run analysis
        report = pipeline.analyze_media(temp_path, is_image=False, update_progress_cb=update_progress)
        
        # Cache completed result
        db_manager.cache_result(file_hash, report)
        
        # Update task status to completed
        db_manager.update_task_status(task_id, {
            "status": "completed",
            "progress": 100,
            "stage": "Finished",
            "task_id": task_id,
            "result": report
        })
    except Exception as e:
        print(f"Error in background video processing task {task_id}: {e}")
        db_manager.update_task_status(task_id, {
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
            db_manager.update_task_status(task_id, {
                "status": "processing",
                "progress": progress_pct,
                "stage": stage_name,
                "task_id": task_id,
                "filename": filename
            })

        report = voice_pipeline.analyze_audio(temp_path, update_progress_cb=update_progress)

        db_manager.cache_result(file_hash, report)

        db_manager.update_task_status(task_id, {
            "status": "completed",
            "progress": 100,
            "stage": "Finished",
            "task_id": task_id,
            "result": report
        })
    except Exception as e:
        print(f"Error in background audio processing task {task_id}: {e}")
        db_manager.update_task_status(task_id, {
            "status": "failed",
            "progress": 0,
            "error": str(e),
            "task_id": task_id
        })
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/detect/audio")
async def detect_voice_deepfake(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Uploads an audio file for voice deepfake detection. Checks cache first;
    always processed asynchronously via background task + polling.
    """
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    is_audio = ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"]

    if not is_audio:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload an audio file (wav, mp3, flac, ogg, m4a, aac).")

    try:
        file_hash = get_file_hash_from_upload(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

    cached_result = db_manager.get_cached_result(file_hash)
    if cached_result:
        print(f"Cache hit for file hash: {file_hash}")
        cached_result["timestamp"] = time_tracker_helper()
        db_manager.cache_result(file_hash, cached_result)
        return {
            "status": "completed",
            "source": "cache",
            "result": cached_result
        }

    task_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{task_id}{ext}")

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    db_manager.update_task_status(task_id, {
        "status": "processing",
        "progress": 10,
        "stage": "Initializing background task",
        "task_id": task_id,
        "filename": filename
    })
    background_tasks.add_task(async_analyze_audio, task_id, temp_path, file_hash, filename)
    return {
        "status": "processing",
        "task_id": task_id,
        "message": "Voice analysis queued in background."
    }

@app.post("/api/detect")
async def detect_deepfake(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...)
):
    """
    Uploads an image or video file for deepfake detection.
    Checks cache first; executes synchronously for images and asynchronously for videos.
    """
    # 1. Check file type
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    is_image = ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    is_video = ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]

    if not is_image and not is_video:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload an image or video.")

    # 2. Get file hash and check cache
    try:
        file_hash = get_file_hash_from_upload(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

    cached_result = db_manager.get_cached_result(file_hash)
    if cached_result:
        print(f"Cache hit for file hash: {file_hash}")
        # Update timestamp to refresh history
        cached_result["timestamp"] = time_tracker_helper()
        db_manager.cache_result(file_hash, cached_result)
        return {
            "status": "completed",
            "source": "cache",
            "result": cached_result
        }

    # 3. Save uploaded file to temp directory
    task_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{task_id}{ext}")
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 4. Process
    if is_image:
        # Images are processed synchronously (usually < 1s)
        try:
            report = pipeline.analyze_media(temp_path, is_image=True)
            db_manager.cache_result(file_hash, report)
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return {
                "status": "completed",
                "source": "fresh",
                "result": report
            }
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    else:
        # Videos are sent to background tasks
        db_manager.update_task_status(task_id, {
            "status": "processing",
            "progress": 10,
            "stage": "Initializing background task",
            "task_id": task_id,
            "filename": filename
        })
        background_tasks.add_task(async_analyze_video, task_id, temp_path, file_hash, filename)
        return {
            "status": "processing",
            "task_id": task_id,
            "message": "Video analysis queued in background."
        }

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """
    Polls the progress status of a background video detection job.
    """
    status = db_manager.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found.")
    return status

@app.get("/api/history")
async def get_history():
    """
    Fetches the history of completed analyses.
    """
    return db_manager.get_history()

@app.post("/api/clear")
async def clear_cache():
    """
    Clears all local cached records.
    """
    db_manager.local_cache.clear()
    db_manager.local_tasks.clear()
    if db_manager.redis_client:
        try:
            db_manager.redis_client.flushdb()
        except Exception as e:
            print(f"Redis flush error: {e}")
    return {"message": "Cache and task history cleared."}

def time_tracker_helper():
    import time
    return time.time()
