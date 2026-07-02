import { useState } from 'react';
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
  const [imgDims, setImgDims] = useState({ w: 1, h: 1 });

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
          <div className="glass-panel" style={{ padding: '1rem', marginBottom: '2rem', textAlign: 'center' }}>
            <h3 style={{ fontSize: '1.1rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
              <ImageIcon size={18} color="var(--primary)" /> Media Preview & Localization
            </h3>
            <div style={{ position: 'relative', display: 'inline-block', width: '100%', maxWidth: '600px' }}>
              {file && (
                <img 
                  src={URL.createObjectURL(file)} 
                  alt="Preview" 
                  onLoad={(e) => setImgDims({ w: e.target.naturalWidth, h: e.target.naturalHeight })} 
                  style={{ width: '100%', height: 'auto', display: 'block', borderRadius: '8px' }} 
                />
              )}
              {faces.map((face, idx) => {
                const [x, y, w, h] = face.box;
                const left = (x / imgDims.w) * 100 + '%';
                const top = (y / imgDims.h) * 100 + '%';
                const width = (w / imgDims.w) * 100 + '%';
                const height = (h / imgDims.h) * 100 + '%';
                const isFake = face.fake_score > 0.5;
                return (
                  <div key={`face-${idx}`} style={{ position: 'absolute', left, top, width, height, border: `3px solid ${isFake ? '#ef4444' : '#22c55e'}`, boxShadow: '0 0 10px rgba(0,0,0,0.5)', borderRadius: '4px' }}>
                    <span style={{ position: 'absolute', top: '-28px', left: '-3px', background: isFake ? '#ef4444' : '#22c55e', color: 'white', padding: '2px 8px', fontSize: '12px', fontWeight: 'bold', borderRadius: '4px', whiteSpace: 'nowrap' }}>
                      {isFake ? 'FAKE FACE' : 'REAL FACE'} {formatPct(face.fake_score)}%
                    </span>
                  </div>
                );
              })}
              
              {/* Render VLM Anomaly Regions */}
              {result.vlm_analysis?.anomaly_regions?.map((region, idx) => {
                const [x, y, w, h] = region.box;
                const left = (x / imgDims.w) * 100 + '%';
                const top = (y / imgDims.h) * 100 + '%';
                const width = (w / imgDims.w) * 100 + '%';
                const height = (h / imgDims.h) * 100 + '%';
                return (
                  <div key={`vlm-${idx}`} style={{ position: 'absolute', left, top, width, height, border: '3px solid #f59e0b', boxShadow: '0 0 10px rgba(0,0,0,0.5)', borderRadius: '4px', pointerEvents: 'none' }}>
                    <span style={{ position: 'absolute', top: '-28px', left: '-3px', background: '#f59e0b', color: '#fff', padding: '2px 8px', fontSize: '12px', fontWeight: 'bold', borderRadius: '4px', whiteSpace: 'nowrap' }}>
                      ⚠️ {region.label}
                    </span>
                  </div>
                );
              })}
            </div>
            
            {result.vlm_analysis && (
              <div style={{ marginTop: '1rem', textAlign: 'left', background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', fontSize: '0.95rem' }}>
                <h4 style={{ marginBottom: '0.5rem', color: 'var(--primary)' }}>Semantic & Physical Analysis</h4>
                <p style={{ color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                  {result.vlm_analysis.reasoning}
                </p>
              </div>
            )}
          </div>

          <ResultGauge
            result={{
              ...result,
              stats: [
                { label: 'Analysis Target', value: <><ImageIcon size={18} color="var(--secondary)" /> Image File</> },
                { label: 'Processing Latency', value: `${result.processing_time_sec}s` },
                { label: 'Faces Detected', value: result.total_faces_detected },
                { label: 'Web Trace Anomaly Score', value: `${formatPct(result.web_score || 0)}%` },
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
                    <div className="face-card-actions" style={{ position: 'absolute', bottom: '0', left: '0', right: '0', padding: '0.5rem', background: 'rgba(0,0,0,0.8)', borderTop: '1px solid rgba(255,255,255,0.1)', transform: 'translateY(100%)', transition: 'transform 0.2s' }}>
                       <span style={{color: 'var(--text-muted)', fontSize: '0.8rem'}}>Background Scan Active</span>
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
