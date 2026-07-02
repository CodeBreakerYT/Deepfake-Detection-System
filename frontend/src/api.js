import { db } from "./firebase";
import { doc, getDoc, setDoc, collection, getDocs, orderBy, query, writeBatch } from "firebase/firestore";

export const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

// Helper to compute SHA-256 hash in JS
export async function getFileHash(file) {
  const arrayBuffer = await file.arrayBuffer();
  const hashBuffer = await crypto.subtle.digest("SHA-256", arrayBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, "0")).join("");
}

// Convert a File object to a base64 data URI string
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Determine media type from the API endpoint
function getMediaType(endpoint, filename) {
  if (endpoint.includes("audio")) return "audio";
  const ext = filename.split(".").pop().toLowerCase();
  const imageExts = ["jpg", "jpeg", "png", "bmp", "webp"];
  return imageExts.includes(ext) ? "image" : "video";
}

// Save the full report + file base64 as a single Firestore document
async function saveToFirestore(fileHash, report, fileBase64) {
  try {
    const record = {
      ...report,
      fileBase64,
    };
    await setDoc(doc(db, "deepfake_results", fileHash), record);
    console.log("Saved full report with media to Firestore for:", report.name);
  } catch (err) {
    console.warn("Failed to write result to Firestore:", err);
  }
}

export async function uploadAndTrack(endpoint, file, { onStage, onResult, onError }) {
  onStage({ status: "uploading", progress: 5, stage: "Checking cache..." });

  try {
    // 1. Get file hash and convert file to base64
    const fileHash = await getFileHash(file);
    const mediaType = getMediaType(endpoint, file.name);
    const fileBase64 = await fileToBase64(file);

    // 2. Check Firestore cache first
    try {
      const docRef = doc(db, "deepfake_results", fileHash);
      const docSnap = await getDoc(docRef);
      if (docSnap.exists()) {
        console.log("Firebase Cache hit for file hash:", fileHash);
        let cachedData = docSnap.data();
        
        // Auto-heal corrupted cache structure from previous backend bug
        if (cachedData.result && typeof cachedData.result === 'object' && cachedData.result.average_heuristics) {
          const nestedResult = cachedData.result;
          cachedData = { ...cachedData, ...nestedResult };
          delete cachedData.result; // remove the nested duplicate to avoid confusion
        }
        
        cachedData.timestamp = Date.now() / 1000;
        await setDoc(docRef, cachedData).catch(err => console.warn("Failed to update timestamp in Firestore:", err));
        onResult(cachedData);
        return;
      }
    } catch (firebaseErr) {
      console.warn("Firestore cache check failed, continuing with backend:", firebaseErr);
    }

    // 3. Upload file directly to Flask backend for deepfake analysis
    onStage({ status: "uploading", progress: 15, stage: "Uploading to analysis server..." });
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${API_BASE}${endpoint}`, { method: "POST", body: formData });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || "Server upload failed");
    }

    const data = await res.json();

    if (data.status === "completed") {
      const report = data.result;
      report.timestamp = Date.now() / 1000;
      report.verdict = report.is_fake ? "FAKE" : "REAL";
      report.name = file.name;
      report.category = report.is_fake ? "fake" : "real";
      report.media = mediaType;

      // Save full report + file base64 as one Firestore document
      await saveToFirestore(fileHash, report, fileBase64);

      onResult(report);
      return;
    }

    if (data.status === "processing") {
      onStage({ status: "processing", progress: data.progress || 10, stage: data.stage || "Processing..." });
      pollStatus(data.task_id, fileHash, file.name, mediaType, fileBase64, { onStage, onResult, onError });
    }
  } catch (err) {
    onError(err.message);
  }
}

function pollStatus(taskId, fileHash, filename, mediaType, fileBase64, { onStage, onResult, onError }) {
  const poll = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/status/${taskId}`);
      if (!res.ok) throw new Error("Failed to get task status");

      const data = await res.json();
      if (data.status === "processing") {
        onStage({ status: "processing", progress: data.progress, stage: data.stage });
        setTimeout(poll, 1200);
      } else if (data.status === "completed") {
        const report = data.result;
        report.timestamp = Date.now() / 1000;
        report.verdict = report.is_fake ? "FAKE" : "REAL";
        report.name = filename;
        report.category = report.is_fake ? "fake" : "real";
        report.media = mediaType;

        // Save full report + file base64 as one Firestore document
        await saveToFirestore(fileHash, report, fileBase64);

        onResult(report);
      } else if (data.status === "failed") {
        onError(data.error || "Processing failed.");
      }
    } catch (err) {
      onError(err.message);
    }
  };
  setTimeout(poll, 800);
}

export async function fetchHistory() {
  // 1. Try to fetch from Firebase Firestore first
  try {
    console.log("Attempting to fetch history from Cloud Firestore...");
    const q = query(collection(db, "deepfake_results"), orderBy("timestamp", "desc"));
    const querySnapshot = await getDocs(q);
    const history = [];
    querySnapshot.forEach((doc) => {
      history.push(doc.data());
    });
    console.log(`Fetched ${history.length} items from Firestore.`);
    return history;
  } catch (err) {
    console.warn("Firestore history fetch failed, falling back to backend history API:", err);
  }

  // 2. Fallback to backend API
  try {
    const res = await fetch(`${API_BASE}/api/history`);
    if (!res.ok) throw new Error("Failed to fetch history from backend");
    return await res.json();
  } catch (backendErr) {
    console.error("Backend history fallback failed:", backendErr);
    return [];
  }
}

export async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    return res.ok;
  } catch {
    return false;
  }
}

export async function clearAllHistory() {
  // 1. Clear from Firestore
  try {
    console.log("Attempting to clear history in Firestore...");
    const querySnapshot = await getDocs(collection(db, "deepfake_results"));
    const batch = writeBatch(db);
    querySnapshot.forEach((doc) => {
      batch.delete(doc.ref);
    });
    await batch.commit();
    console.log("Firestore history cleared.");
  } catch (err) {
    console.warn("Failed to clear Firestore history:", err);
  }

  // 2. Clear backend local cache
  try {
    await fetch(`${API_BASE}/api/clear`, { method: "POST" });
  } catch (err) {
    console.error("Failed to clear backend cache:", err);
  }
}

export async function searchWebForMatches(fileOrBase64) {
  try {
    let res;
    if (fileOrBase64 instanceof File || fileOrBase64 instanceof Blob) {
      const formData = new FormData();
      formData.append("file", fileOrBase64);
      res = await fetch(`${API_BASE}/api/search-web`, {
        method: "POST",
        body: formData,
      });
    } else if (typeof fileOrBase64 === "string") {
      res = await fetch(`${API_BASE}/api/search-web`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: fileOrBase64 }),
      });
    } else {
      throw new Error("Invalid input for web search");
    }

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || "Failed to search web");
    }

    const data = await res.json();
    return data.lens_url;
  } catch (err) {
    console.error("Error searching web:", err);
    throw err;
  }
}
