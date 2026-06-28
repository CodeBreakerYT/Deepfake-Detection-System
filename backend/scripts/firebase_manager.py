import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

class FirebaseManager:
    """
    Manages connection to Firebase Firestore for caching deepfake detection results
    and tracking async task states. Falls back to in-memory dictionaries if Firebase is offline.
    Uses a single collection 'deepfake_analyses' for all documents.
    """
    def __init__(self):
        self.db = None
        self.local_cache = {}

        cred_json_str = os.environ.get("FIREBASE_CREDENTIALS_JSON")
        cred_file_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
        cred = None

        try:
            if cred_json_str:
                cred_info = json.loads(cred_json_str)
                cred = credentials.Certificate(cred_info)
            elif cred_file_path and os.path.exists(cred_file_path):
                cred = credentials.Certificate(cred_file_path)

            if not firebase_admin._apps:
                if cred:
                    firebase_admin.initialize_app(cred)
                else:
                    firebase_admin.initialize_app()

            self.db = firestore.client()
            print("Connected to Firebase Firestore successfully!")
        except Exception as e:
            print(f"Warning: Firebase is not reachable ({e}). Falling back to in-memory cache.")
            self.db = None

    def get_cached_result(self, file_hash: str) -> dict | None:
        if self.db:
            try:
                doc_ref = self.db.collection("deepfake_analyses").document(file_hash)
                doc = doc_ref.get()
                if doc.exists:
                    return doc.to_dict()
            except Exception as e:
                print(f"Firebase get cache error: {e}")
        return self.local_cache.get(file_hash)

    def cache_result(self, file_hash: str, doc_data: dict) -> None:
        self.local_cache[file_hash] = doc_data
        if self.db:
            try:
                doc_ref = self.db.collection("deepfake_analyses").document(file_hash)
                doc_ref.set(doc_data, merge=True)
            except Exception as e:
                print(f"Firebase set cache error: {e}")

    def get_task_status(self, task_id: str) -> dict | None:
        if self.db:
            try:
                docs = self.db.collection("deepfake_analyses").where("task_id", "==", task_id).limit(1).get()
                if docs:
                    return docs[0].to_dict()
            except Exception as e:
                print(f"Firebase get task error: {e}")
        
        # Local search
        for k, v in self.local_cache.items():
            if v.get("task_id") == task_id:
                return v
        return None

    def update_task_status(self, file_hash: str, status: dict) -> None:
        existing = self.local_cache.get(file_hash, {})
        existing.update(status)
        self.local_cache[file_hash] = existing
        
        if self.db:
            try:
                doc_ref = self.db.collection("deepfake_analyses").document(file_hash)
                doc_ref.set(status, merge=True)
            except Exception as e:
                print(f"Firebase set task error: {e}")

    def get_history(self) -> list[dict]:
        history = []
        if self.db:
            try:
                docs = self.db.collection("deepfake_analyses").stream()
                for doc in docs:
                    res = doc.to_dict()
                    if res and "result" in res:
                        history.append(self._strip_heavy_payload(res))
                history.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
                return history
            except Exception as e:
                print(f"Firebase get history error: {e}")
                
        for file_hash, res in self.local_cache.items():
            if "result" in res:
                history.append(self._strip_heavy_payload(res))
        history.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return history

    def _strip_heavy_payload(self, res: dict) -> dict:
        light_res = res.copy()
        if "image" in light_res:
            light_res["image"] = None
        if "result" in light_res and isinstance(light_res["result"], dict):
            if "frames" in light_res["result"]:
                for frame in light_res["result"]["frames"]:
                    if "image" in frame:
                        frame["image"] = None
                    if "faces" in frame:
                        for face in frame["faces"]:
                            if "crop_b64" in face:
                                face["crop_b64"] = None
        return light_res

    def clear_cache(self) -> None:
        self.local_cache.clear()
        if self.db:
            try:
                results_ref = self.db.collection("deepfake_analyses")
                docs = results_ref.list_documents()
                for doc in docs:
                    doc.delete()
                print("Firebase cache cleared.")
            except Exception as e:
                print(f"Firebase clear cache error: {e}")
