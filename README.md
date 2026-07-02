# DeepShield — Deepfake Detection System

Detects AI-generated/deepfake content in images, video, and voice audio. Flask
backend (an ensemble of pretrained transformer models — face-forgery ViT +
general AI-image Swin detector for images/video, Wav2Vec2 for voice — blended
with classical pixel/acoustic forensics) with a React frontend. No login
required.

## Local development

**Backend** (from `backend/`):
```
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python app.py
```
Runs on `http://localhost:8000`. First run downloads three Hugging Face
models (~1.6GB total: two image detectors + one audio detector) into
`backend/hf_cache/`; cached after that.

**Frontend** (from `frontend/`):
```
npm install
npm run dev
```
Open `http://localhost:5173`.

## Deploying

The backend (PyTorch, transformers, OpenCV, librosa, background task
threads, multi-GB model weights) needs a persistent container, not a
serverless function, so the deployment is split:

- **Frontend → Netlify**
- **Backend → Hugging Face Spaces** (Docker SDK)

### Backend on Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space):
   SDK = **Docker**, visibility = your choice.
2. Push the contents of `backend/` to the Space's git repo root (the Space
   *is* a git repo — its own remote, separate from GitHub):
   ```
   cd backend
   git init
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git add -A
   git commit -m "Deploy backend"
   git push space main
   ```
   (If `backend/` is already tracked inside this repo's git history and you'd
   rather push directly from the monorepo, use `git subtree push --prefix
   backend space main` from the repo root instead.)
3. The Space's `Dockerfile` pre-downloads all three models at build time, so
   the first build takes a while (large image) but cold starts afterward
   don't re-download anything.
4. In the Space's **Settings → Variables and secrets**, set:
   - `FRONTEND_ORIGINS` — your Netlify URL, e.g. `https://deepshield.netlify.app`
     (comma-separated if multiple; defaults to `*` if unset).
   - `HF_TOKEN` *(optional)* — avoids anonymous Hugging Face Hub rate limits.
   - `FIREBASE_CREDENTIALS_JSON` *(optional)* — raw JSON of a Firebase service
     account for persistent history/caching (otherwise falls back to in-memory,
     which resets whenever the Space restarts/sleeps).
5. Your API base URL will be `https://<your-username>-<space-name>.hf.space`.
6. **Free tier**: CPU Basic (16GB RAM) comfortably fits all three models.
   Spaces sleep after inactivity — the first request after waking takes a
   few extra seconds to reload the models into memory (not re-download,
   since they're baked into the image).

### Frontend on Netlify

1. Import the repo at [app.netlify.com](https://app.netlify.com/start), set
   **Base directory** to `frontend`.
2. Build command `npm run build`, publish directory `frontend/dist`
   (already configured in `frontend/netlify.toml`).
3. Set environment variable `VITE_API_BASE` to your Space URL from above,
   e.g. `https://your-username-deepshield-backend.hf.space`.
4. `frontend/netlify.toml` already rewrites all routes to `index.html` so
   client-side routing (`/image`, `/video`, `/voices`, `/about`) works on
   direct navigation/refresh.
5. Firebase config is optional client-side (falls back to a baked-in demo
   project) — set `VITE_FIREBASE_*` env vars if you want to point at your
   own Firestore project instead.
