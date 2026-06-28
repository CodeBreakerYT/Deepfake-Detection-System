import os
import json
import redis

class RedisManager:
    """
    Manages connection to Redis for caching deepfake detection results
    and tracking async task states. Falls back to in-memory dictionaries if Redis is offline.
    """
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        self.redis_client = None
        self.local_cache = {}
        self.local_tasks = {}

        # Render/managed Redis providers expose a single connection string (REDIS_URL)
        # instead of separate host/port; prefer it when present.
        redis_url = os.environ.get("REDIS_URL")

        if not self._is_redis_reachable(redis_url, host, port):
            print("Warning: Redis is not reachable. Falling back to in-memory cache.")
            self.redis_client = None
            return

        try:
            if redis_url:
                self.redis_client = redis.from_url(
                    redis_url, socket_connect_timeout=2.0, decode_responses=True
                )
            else:
                # Short timeout to avoid blocking startup if Redis is down
                self.redis_client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    socket_connect_timeout=2.0,
                    decode_responses=True
                )
            # Test connection
            self.redis_client.ping()
            print(f"Connected to Redis at {redis_url or f'{host}:{port}'}")
        except (redis.ConnectionError, redis.TimeoutError):
            print(f"Warning: Redis is not reachable. Falling back to in-memory cache.")
            self.redis_client = None

    def _is_redis_reachable(self, redis_url: str | None, host: str, port: int) -> bool:
        """
        Quickly check if the Redis port is listening using a basic socket connection
        to avoid library-specific blocking hangs during DNS resolution or connect.
        """
        import socket
        from urllib.parse import urlparse

        target_host = host
        target_port = port

        if redis_url:
            try:
                parsed = urlparse(redis_url)
                target_host = parsed.hostname or host
                target_port = parsed.port or port
            except Exception:
                pass

        try:
            # Use a quick 1.0 second connection timeout to see if port is open
            s = socket.create_connection((target_host, target_port), timeout=1.0)
            s.close()
            return True
        except Exception:
            return False

    def get_cached_result(self, file_hash: str) -> dict | None:
        """
        Retrieves cached result by file hash.
        """
        key = f"deepfake:result:{file_hash}"
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                print(f"Redis get cache error: {e}")
        return self.local_cache.get(file_hash)

    def cache_result(self, file_hash: str, result: dict, expire_sec: int = 86400) -> None:
        """
        Caches detection result.
        """
        key = f"deepfake:result:{file_hash}"
        if self.redis_client:
            try:
                self.redis_client.setex(key, expire_sec, json.dumps(result))
                # Add to a history set for easy listing
                self.redis_client.sadd("deepfake:history:hashes", file_hash)
            except Exception as e:
                print(f"Redis set cache error: {e}")
        self.local_cache[file_hash] = result

    def get_task_status(self, task_id: str) -> dict | None:
        """
        Retrieves task status.
        """
        key = f"deepfake:task:{task_id}"
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                print(f"Redis get task error: {e}")
        return self.local_tasks.get(task_id)

    def update_task_status(self, task_id: str, status: dict, expire_sec: int = 3600) -> None:
        """
        Updates background task status.
        """
        key = f"deepfake:task:{task_id}"
        if self.redis_client:
            try:
                self.redis_client.setex(key, expire_sec, json.dumps(status))
            except Exception as e:
                print(f"Redis set task error: {e}")
        self.local_tasks[task_id] = status

    def get_history(self) -> list[dict]:
        """
        Gets a history list of all completed analyses.
        """
        history = []
        if self.redis_client:
            try:
                hashes = self.redis_client.smembers("deepfake:history:hashes")
                for file_hash in hashes:
                    res = self.get_cached_result(file_hash)
                    if res:
                        # Strip raw base64 images from history list to keep response lightweight
                        light_res = res.copy()
                        if "frames" in light_res:
                            for frame in light_res["frames"]:
                                if "image" in frame:
                                    frame["image"] = None  # Remove heavy image payload
                                if "faces" in frame:
                                    for face in frame["faces"]:
                                        if "crop_b64" in face:
                                            face["crop_b64"] = None
                        history.append(light_res)
                # Sort by timestamp, newest first
                history.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
                return history
            except Exception as e:
                print(f"Redis get history error: {e}")
                
        # Fallback to local memory history
        for file_hash, res in self.local_cache.items():
            light_res = res.copy()
            if "frames" in light_res:
                for frame in light_res["frames"]:
                    if "image" in frame:
                        frame["image"] = None
                    if "faces" in frame:
                        for face in frame["faces"]:
                            if "crop_b64" in face:
                                face["crop_b64"] = None
            history.append(light_res)
        history.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return history
