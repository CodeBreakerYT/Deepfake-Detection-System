import { Image as ImageIcon, Cpu, Camera } from 'lucide-react';
import UploadZone from '../components/UploadZone';
import ProgressPanel from '../components/ProgressPanel';
import ErrorPanel from '../components/ErrorPanel';
import ResultGauge from '../components/ResultGauge';
import { formatPct } from '../utils';
import HeuristicBars from '../components/HeuristicBars';
import { useDetection } from '../hooks/useDetection';

export default function ImagePage() {
  const { file, status, progress, stage, result, error, processUpload, reset } = useDetection('/api/detect');

  const faces = result?.frames?.[0]?.faces || [];

  return (
    <main className="main-panel">
      <div className="page-intro">
        <h1>Image Deepfake Analysis</h1>
        <p>Upload a photo to scan for face-swap artifacts, AI-generation traces, and pixel-level forensic anomalies.</p>
      </div>

      {status === null && (
        <UploadZone
          accept="image/*"
          title="Submit an Image for Verification"
          subtitle="Drag and drop a JPG, PNG, BMP, or WEBP file here, or click to browse."
          onFile={processUpload}
        />
      )}

      {(status === 'uploading' || status === 'processing') && (
        <ProgressPanel
          filename={file?.name}
          stage={stage}
          progress={progress}
          hint="Running facial localization and forensic model inference..."
        />
      )}

      {status === 'failed' && <ErrorPanel message={error} onRetry={reset} />}

      {status === 'completed' && result && (
        <>
          <ResultGauge
            result={{
              ...result,
              stats: [
                { label: 'Analysis Target', value: <><ImageIcon size={18} color="var(--secondary)" /> Image File</> },
                { label: 'Processing Latency', value: `${result.processing_time_sec}s` },
                { label: 'Faces Detected', value: result.total_faces_detected },
                { label: 'Verdict Confidence', value: `${formatPct(result.confidence)}%` },
              ],
            }}
            icon={<Cpu size={14} />}
            modeLine={result.used_vit_model ? 'Transformer Model Active (ViT_Deepfake_Detection)' : 'Local Heuristics & Image Forensics Active'}
            onReset={reset}
          />

          <HeuristicBars
            title="Pixel-Level Visual Artifact Markers"
            items={[
              { label: 'Boundary Blurring', value: result.average_heuristics.blur_artifact_score },
              { label: 'Spectral Frequency Anomalies', value: result.average_heuristics.frequency_anomaly_score },
              { label: 'Lighting & Skin Tone Outliers', value: result.average_heuristics.color_anomaly_score },
            ]}
          />

          {faces.length > 0 && (
            <div className="glass-panel">
              <div className="faces-title">
                <Camera size={18} color="var(--primary)" />
                <span>Detected Face Profiles ({faces.length})</span>
              </div>
              <div className="faces-grid">
                {faces.map((face, idx) => (
                  <div className="face-card" key={idx}>
                    <img src={`data:image/jpeg;base64,${face.crop_b64}`} className="face-crop-img" alt="Face Crop" />
                    <div className="face-info-overlay">
                      <div className={`face-card-score ${face.fake_score > 0.5 ? 'danger' : 'safe'}`}>
                        {formatPct(face.fake_score)}% Fake
                      </div>
                      <div className="face-card-meta">Confidence {formatPct(face.confidence)}%</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {faces.length === 0 && (
            <div className="glass-panel" style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem' }}>
              <ImageIcon size={28} style={{ marginBottom: '0.5rem', opacity: 0.5 }} />
              <p>No faces were detected in this image.</p>
            </div>
          )}
        </>
      )}
    </main>
  );
}
