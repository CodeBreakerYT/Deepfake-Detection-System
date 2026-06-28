import os
import json
import threading

class LocalStore:
    """
    Simple JSON-file-backed cache for the desktop app. Replaces RedisManager
    for the single-user, no-server desktop use case - persists analysis
    results across app restarts without needing any external service.
    """
    def __init__(self, store_path: str = "deepshield_history.json"):
        self.store_path = store_path
        self._lock = threading.Lock()
        self.cache = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save(self) -> None:
        try:
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f)
        except OSError as e:
            print(f"Warning: failed to persist history to {self.store_path}: {e}")

    def get_cached_result(self, file_hash: str) -> dict | None:
        return self.cache.get(file_hash)

    def cache_result(self, file_hash: str, result: dict) -> None:
        with self._lock:
            self.cache[file_hash] = result
            self._save()

    def get_history(self) -> list[dict]:
        history = list(self.cache.values())
        history.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return history

    def clear(self) -> None:
        with self._lock:
            self.cache = {}
            self._save()
