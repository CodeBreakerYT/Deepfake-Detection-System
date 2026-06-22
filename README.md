# DeepShield — Deepfake Detection System

Detects deepfakes in images, video, and voice audio. FastAPI backend (PyTorch ViT model + OpenCV/FFT forensics for images/video, Wav2Vec2 model + acoustic forensics for voice) with a React frontend. No login required.

## Local development

**Backend** (from repo root):
```
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
First run downloads two Hugging Face models (~350MB ViT + ~1.2GB Wav2Vec2). If the voice model fails to download, the app automatically falls back to a heuristic-only acoustic classifier — it still works, just less accurately for voice.

**Frontend**:
```
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173`.

## Deploying

The backend (PyTorch, transformers, OpenCV, librosa, background task polling) is too heavy and stateful for Vercel's serverless functions, so the deployment is split:

- **Frontend → Vercel**
- **Backend → Render** (or Railway — anything that runs a persistent Python process)

### Backend on Render

1. Push this repo to GitHub, connect it on [render.com](https://render.com).
2. Render will detect `render.yaml` and provision a Python web service automatically (Blueprint deploy), or set up manually:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Set the `FRONTEND_ORIGINS` env var to your Vercel URL once you have it (comma-separated if multiple), e.g. `https://deepshield.vercel.app`.
4. Optional: add a Render Redis instance and set `REDIS_URL` on the web service for persistent result caching across restarts (without it, caching falls back to in-memory and resets on redeploy/restart).
5. **RAM**: Render's free tier (512MB) is tight for PyTorch + two loaded models — if you hit OOM/crashes, upgrade to at least the Starter plan.
6. Free-tier services spin down when idle and re-download the HF models on the next request — first request after idling will be slow.

### Frontend on Vercel

1. Import the repo on [vercel.com](https://vercel.com), set **Root Directory** to `frontend`.
2. Build command `npm run build`, output directory `dist` (Vercel autodetects this for Vite).
3. Set environment variable `VITE_API_BASE` to your Render backend URL, e.g. `https://deepshield-backend.onrender.com`.
4. `frontend/vercel.json` is already configured to rewrite all routes to `index.html` so client-side routing (`/image`, `/video`, `/voices`, `/about`) works on direct navigation/refresh.
