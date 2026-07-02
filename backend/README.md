---
title: DeepShield Detection API
emoji: 🛡️
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# DeepShield — Deepfake Detection API

Flask backend for the DeepShield media-forensics dashboard. Classifies images,
video, and audio as real or AI-generated using an ensemble of pretrained
transformer models:

- **Image/video**: `prithivMLmods/Deep-Fake-Detector-v2-Model` (face-forgery ViT)
  + `Organika/sdxl-detector` (general diffusion-image detector), blended with
  pixel-level forensic heuristics.
- **Audio**: `garystafford/wav2vec2-deepfake-voice-detector`, blended with
  acoustic forensic heuristics.

## API

- `POST /api/detect` — image/video upload, returns a deepfake report.
- `POST /api/detect/audio` — audio upload, returns a voice-deepfake report.
- `GET /api/status/<task_id>` — poll progress for async video/audio jobs.
- `GET /api/history` — recent analysis history.
- `GET /api/health` — health check.

## Configuration

Set these as Space secrets/variables (Settings → Variables and secrets):

- `FRONTEND_ORIGINS` — comma-separated allowed CORS origins, e.g. your
  Netlify URL. Defaults to `*` if unset.
- `HF_TOKEN` — optional, avoids anonymous Hugging Face Hub rate limits.
- `FIREBASE_CREDENTIALS_JSON` — optional, enables persistent Firestore
  history/caching instead of the in-memory fallback.

This Space is built from the repository's `backend/` directory — the
`Dockerfile` pre-downloads all three models at build time so cold starts
don't depend on the HF Hub being reachable.
