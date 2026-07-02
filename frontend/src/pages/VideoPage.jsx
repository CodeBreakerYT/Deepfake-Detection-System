import { Video, Cpu, Camera } from 'lucide-react';
import UploadZone from '../components/UploadZone';
import ProgressPanel from '../components/ProgressPanel';
import ErrorPanel from '../components/ErrorPanel';
import ResultGauge from '../components/ResultGauge';
import { formatPct } from '../utils';
import HeuristicBars from '../components/HeuristicBars';
import Timeline from '../components/Timeline';
import { useDetection } from '../hooks/useDetection';

export default function VideoPage() {
  const { file, status, progress, stage, result, error, selectedIdx, setSelectedIdx, processUpload, reset } =
    useDetection('/api/detect');

  const getAllFaces = () => {
    if (!result || !result.frames) return [];
    const faces = [];
    result.frames.forEach((frame) => {
      frame.faces.forEach((face) => faces.push({ ...face, frame_idx: frame.frame_idx, timestamp: frame.timestamp }));
    });
    return faces;
  };

  const allFaces = getAllFaces();
  const currentFrame = result?.frames?.find((f) => f.frame_idx === selectedIdx) || result?.frames?.[0];

  const timelinePoints = result?.frames?.map((f) => ({
    idx: f.frame_idx,
    score: f.faces.length > 0 ? Math.max(...f.faces.map((face) => face.fake_score)) : 0,
  }));

  return (
    <main className="main-panel">
      <div className="page-intro">
        <h1>Video Deepfake Analysis</h1>
        <p>Upload a clip to sample frames, track faces over time, and flag temporal inconsistencies frame by frame.</p>
      </div>

      {status === null && (
        <UploadZone
          accept="video/*"
          title="Submit a Video for Verification"
          subtitle="Drag and drop an MP4, AVI, MOV, MKV, or WEBM file here, or click to browse."
          onFile={processUpload}
        />
      )}

      {(status === 'uploading' || status === 'processing') && (
        <ProgressPanel
          filename={file?.name}
          stage={stage}
          progress={progress}
          hint="Extracting frames and running facial localization + model inference..."
        />
      )}

      {status === 'failed' && <ErrorPanel message={error} onRetry={reset} />}

      {status === 'completed' && result && (
        <>
          <div className="glass-panel" style={{ padding: '1rem', marginBottom: '2rem', textAlign: 'center' }}>
            <h3 style={{ fontSize: '1.1rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
              <Video size={18} color="var(--primary)" /> Video Playback Preview
            </h3>
            <div style={{ position: 'relative', display: 'inline-block', width: '100%', maxWidth: '600px' }}>
              {file && (
                <video 
                  src={URL.createObjectURL(file)} 
                  controls
                  style={{ width: '100%', height: 'auto', display: 'block', borderRadius: '8px', border: '1px solid var(--border)' }} 
                />
              )}
            </div>
          </div>

          <ResultGauge
            result={{
              ...result,
              stats: [
                { label: 'Analysis Target', value: <><Video size={18} color="var(--secondary)" /> Video File</> },
                { label: 'Processing Latency', value: `${result.processing_time_sec}s` },
                { label: 'Sampled Frames', value: result.total_frames_analyzed },
                { label: 'Total Faces Detected', value: result.total_faces_detected },
                { label: 'Web Trace Anomaly Score', value: `${formatPct(result.web_score || 0)}%` },
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

          {timelinePoints && timelinePoints.length > 1 && (
            <>
              <Timeline
                title="Temporal Score Analysis (Select points to view frame details)"
                points={timelinePoints}
                selectedIdx={selectedIdx}
                onSelect={setSelectedIdx}
              />

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '1.5rem' }}>
                <div className="glass-panel">
                  <h4 style={{ fontSize: '0.9rem', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
                    <Camera size={14} color="var(--primary)" /> Selected Frame Inspector (Frame #{currentFrame?.frame_idx + 1})
                  </h4>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    Timestamp: <strong style={{ color: '#fff' }}>{currentFrame?.timestamp}s</strong> •
                    Detected Faces: <strong style={{ color: '#fff' }}>{currentFrame?.faces?.length || 0}</strong>
                  </p>

                  {currentFrame?.faces?.length > 0 ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '1rem' }}>
                      {currentFrame.faces.map((face) => (
                        <div key={face.face_id} style={{ display: 'flex', alignItems: 'center', gap: '1rem', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)', borderRadius: '8px', padding: '0.5rem 0.75rem' }}>
                          <img src={`data:image/jpeg;base64,${face.crop_b64}`} alt="Face Crop" style={{ width: '45px', height: '45px', borderRadius: '6px', objectFit: 'cover', border: '1px solid rgba(255,255,255,0.1)' }} />
                          <div style={{ flex: 1 }}>
                            <div style={{ fontSize: '0.8rem', fontWeight: 600 }}>Face #{face.face_id + 1}</div>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Bounding Box: [{face.box.join(', ')}]</div>
                          </div>
                          <div style={{ textAlign: 'right' }}>
                            <div className={`face-card-score ${face.fake_score > 0.5 ? 'danger' : 'safe'}`}>
                              {formatPct(face.fake_score)}% Fake
                            </div>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>Confidence: {formatPct(face.confidence)}%</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem', fontSize: '0.8rem' }}>
                      No faces detected on this frame.
                    </div>
                  )}
                </div>

                <div className="glass-panel">
                  <h4 style={{ fontSize: '0.9rem', marginBottom: '0.75rem', fontWeight: 600 }}>Frame Spectral Metrics</h4>
                  {currentFrame?.faces?.length > 0 ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.03)', paddingBottom: '0.25rem' }}>
                        <span>Sharpness Val:</span>
                        <span style={{ color: '#fff', fontWeight: 500 }}>{currentFrame.faces[0].heuristics.sharpness_val}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.03)', paddingBottom: '0.25rem' }}>
                        <span>Freq Ratio:</span>
                        <span style={{ color: '#fff', fontWeight: 500 }}>{currentFrame.faces[0].heuristics.freq_ratio}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>Color Depth Val:</span>
                        <span style={{ color: '#fff', fontWeight: 500 }}>{currentFrame.faces[0].heuristics.color_depth_val}</span>
                      </div>
                      </div>
                  ) : (
                    <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem', fontSize: '0.8rem' }}>
                      No frame metrics available.
                    </div>
                  )}
                </div>
              </div>
            </>
          )}

          {allFaces.length > 0 && (
            <div className="glass-panel">
              <div className="faces-title">
                <Camera size={18} color="var(--primary)" />
                <span>Detected Face Profiles Library ({allFaces.length} total)</span>
              </div>
              <div className="faces-grid">
                {allFaces.map((face, idx) => (
                  <div className="face-card" key={idx} onClick={() => setSelectedIdx(face.frame_idx)}>
                    <img src={`data:image/jpeg;base64,${face.crop_b64}`} className="face-crop-img" alt="Face Crop" />
                    <div className="face-info-overlay">
                      <div className={`face-card-score ${face.fake_score > 0.5 ? 'danger' : 'safe'}`}>
                        {formatPct(face.fake_score)}% Fake
                      </div>
                      <div className="face-card-meta">@ {face.timestamp}s</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </main>
  );
}
