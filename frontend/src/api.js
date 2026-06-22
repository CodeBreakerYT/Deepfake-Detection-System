export const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

/**
 * Uploads a file to the given detect endpoint, then polls /api/status/{task_id}
 * if the server queues it as a background job. Calls back into React state setters.
 */
export async function uploadAndTrack(endpoint, file, { onStage, onResult, onError }) {
  onStage({ status: "uploading", progress: 5, stage: "Uploading to analysis server..." });

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}${endpoint}`, { method: "POST", body: formData });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || "Server upload failed");
    }

    const data = await res.json();

    if (data.status === "completed") {
      onResult(data.result);
      return;
    }

    if (data.status === "processing") {
      onStage({ status: "processing", progress: data.progress || 10, stage: data.stage || "Processing..." });
      pollStatus(data.task_id, { onStage, onResult, onError });
    }
  } catch (err) {
    onError(err.message);
  }
}

function pollStatus(taskId, { onStage, onResult, onError }) {
  const poll = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/status/${taskId}`);
      if (!res.ok) throw new Error("Failed to get task status");

      const data = await res.json();
      if (data.status === "processing") {
        onStage({ status: "processing", progress: data.progress, stage: data.stage });
        setTimeout(poll, 1200);
      } else if (data.status === "completed") {
        onResult(data.result);
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
  const res = await fetch(`${API_BASE}/api/history`);
  if (!res.ok) throw new Error("Failed to fetch history");
  return res.json();
}

export async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/history`);
    return res.ok;
  } catch {
    return false;
  }
}

export async function clearAllHistory() {
  await fetch(`${API_BASE}/api/clear`, { method: "POST" });
}
