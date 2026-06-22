import { ShieldAlert, Cpu, Layers, Mic, AlertTriangle } from 'lucide-react';

export default function About() {
  return (
    <main className="main-panel">
      <div className="page-intro">
        <h1>About DeepShield</h1>
        <p>How the detection pipeline works, and what it can and can't tell you.</p>
      </div>

      <div className="glass-panel">
        <div className="heuristics-title">
          <Cpu size={18} color="var(--primary)" />
          <span>Detection Architecture</span>
        </div>
        <p className="about-text">
          Every upload runs through a two-layer pipeline. First, a <strong>deep learning classifier</strong> scores
          the media: a fine-tuned Vision Transformer (<code>Wvolf/ViT_Deepfake_Detection</code>) for faces in images
          and video frames, and a fine-tuned Wav2Vec2 model (<code>wav2vec2-deepfake-voice-detector</code>) for
          speech audio. Second, classical <strong>forensic heuristics</strong> run alongside the model and act as a
          fallback if a model is unavailable: Laplacian blur variance, FFT spectral-ratio anomalies, and color-channel
          deviation for visual media; pitch jitter, spectral flatness, and energy-variation analysis for audio.
        </p>
      </div>

      <div className="about-grid">
        <div className="glass-panel">
          <div className="heuristics-title">
            <Layers size={18} color="var(--secondary)" />
            <span>Image &amp; Video</span>
          </div>
          <ul className="about-list">
            <li>Faces are located with OpenCV Haar cascades and cropped before classification.</li>
            <li>Videos are sampled at up to 30 evenly-spaced frames; every face per frame is scored independently.</li>
            <li>The global score blends the single worst face score (70%) with the average across all faces (30%) — one convincing fake face is enough to flag a video.</li>
          </ul>
        </div>

        <div className="glass-panel">
          <div className="heuristics-title">
            <Mic size={18} color="var(--secondary)" />
            <span>Voice</span>
          </div>
          <ul className="about-list">
            <li>Audio is resampled to 16kHz mono and split into overlapping 3-second windows.</li>
            <li>Each window is scored independently, producing a per-segment timeline similar to video frames.</li>
            <li>Acoustic heuristics target known synthetic-speech tells: unnaturally stable or erratic pitch, flat vocoder-like spectra, and uniform energy envelopes.</li>
          </ul>
        </div>
      </div>

      <div className="glass-panel" style={{ borderColor: 'var(--warning)' }}>
        <div className="heuristics-title">
          <AlertTriangle size={18} color="var(--warning)" />
          <span>Limitations &amp; Disclaimer</span>
        </div>
        <p className="about-text">
          No deepfake detector is perfect — scores are probabilistic signals, not legal proof. Heavily compressed,
          low-resolution, or adversarially-processed media can fool any model. Treat results as one input among
          several when verifying media authenticity, especially for high-stakes decisions.
        </p>
      </div>

      <div className="glass-panel">
        <div className="heuristics-title">
          <ShieldAlert size={18} color="var(--primary)" />
          <span>No Login, No Tracking</span>
        </div>
        <p className="about-text">
          DeepShield does not require an account. Uploaded files are deleted from the server immediately after
          analysis; only the resulting forensic report is cached (keyed by file hash) to make repeat checks instant.
        </p>
      </div>
    </main>
  );
}
