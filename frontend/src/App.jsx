import React, { useState, useEffect, useRef } from 'react';
import { 
  ShieldAlert, 
  ShieldCheck, 
  UploadCloud, 
  Cpu, 
  Video, 
  Image as ImageIcon, 
  Clock, 
  RefreshCw, 
  AlertCircle, 
  TrendingUp, 
  Camera, 
  Layers, 
  HardDrive 
} from 'lucide-react';

const API_BASE = "http://localhost:8000";

function App() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState(null); // 'uploading' | 'processing' | 'completed' | 'failed' | null
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState('');
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [selectedFrameIdx, setSelectedFrameIdx] = useState(0);
  
  const fileInputRef = useRef(null);
  const pollTimerRef = useRef(null);

  // Check backend health & Load history
  useEffect(() => {
    fetchHistory();
    const interval = setInterval(checkHealth, 5000);
    checkHealth();
    return () => {
      clearInterval(interval);
      if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
    };
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/history`);
      if (res.ok) {
        setBackendOnline(true);
      }
    } catch {
      setBackendOnline(false);
    }
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/history`);
      if (res.ok) {
        const data = await res.json();
        setHistory(data);
      }
    } catch (err) {
      console.error("Failed to fetch history:", err);
    }
  };

  const clearHistory = async () => {
    try {
      await fetch(`${API_BASE}/api/clear`, { method: 'POST' });
      setHistory([]);
      if (result && result.file_hash) {
        setResult(null);
        setStatus(null);
      }
    } catch (err) {
      console.error("Failed to clear history:", err);
    }
  };

  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processUpload(e.target.files[0]);
    }
  };

  const triggerFileSelect = () => {
    fileInputRef.current.click();
  };

  const processUpload = async (selectedFile) => {
    setFile(selectedFile);
    setStatus('uploading');
    setProgress(5);
    setStage('Uploading to analysis server...');
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch(`${API_BASE}/api/detect`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Server upload failed");
      }

      const data = await res.json();
      
      if (data.status === 'completed') {
        // Cached or sync image run
        setResult(data.result);
        setStatus('completed');
        setProgress(100);
        setSelectedFrameIdx(0);
        fetchHistory();
      } else if (data.status === 'processing') {
        // Video queued, start polling
        setStatus('processing');
        setProgress(data.progress || 10);
        setStage(data.stage || 'Video analysis in progress...');
        startPolling(data.task_id);
      }
    } catch (err) {
      setError(err.message);
      setStatus('failed');
    }
  };

  const startPolling = (taskId) => {
    if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
    
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/status/${taskId}`);
        if (!res.ok) throw new Error("Failed to get task status");
        
        const data = await res.json();
        if (data.status === 'processing') {
          setProgress(data.progress);
          setStage(data.stage);
          pollTimerRef.current = setTimeout(poll, 1500);
        } else if (data.status === 'completed') {
          setResult(data.result);
          setStatus('completed');
          setProgress(100);
          setSelectedFrameIdx(0);
          fetchHistory();
        } else if (data.status === 'failed') {
          setError(data.error || "Async processing failed.");
          setStatus('failed');
        }
      } catch (err) {
        setError(err.message);
        setStatus('failed');
      }
    };
    
    pollTimerRef.current = setTimeout(poll, 1000);
  };

  const loadHistoryItem = (item) => {
    setResult(item);
    setFile({ name: item.filename });
    setStatus('completed');
    setSelectedFrameIdx(0);
  };

  // Helper formatting values
  const formatPercentage = (val) => `${Math.round(val * 100)}`;
  const formatTime = (ts) => {
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  // Heuristic color helpers
  const getHeuristicColor = (score) => {
    if (score > 0.6) return 'var(--danger)';
    if (score > 0.3) return 'var(--warning)';
    return 'var(--success)';
  };

  // Extract all faces from all frames for the faces gallery
  const getAllDetectedFaces = () => {
    if (!result || !result.frames) return [];
    const faces = [];
    result.frames.forEach((frame) => {
      frame.faces.forEach((face) => {
        faces.push({
          ...face,
          frame_idx: frame.frame_idx,
          timestamp: frame.timestamp
        });
      });
    });
    return faces;
  };

  const allFaces = getAllDetectedFaces();
  const currentFrameData = result?.frames?.find(f => f.frame_idx === selectedFrameIdx) || result?.frames?.[0];

  return (
    <div className="container">
      {/* Header */}
      <header className="dashboard-header">
        <div className="brand">
          <div className="brand-logo">
            <ShieldAlert size={22} />
          </div>
          <div className="brand-title">
            <h1>DeepShield</h1>
            <span>Media Forensics Dashboard</span>
          </div>
        </div>
        
        <div className="system-status">
          <div className="status-badge">
            <span className={`status-indicator ${backendOnline ? 'online' : 'offline'}`}></span>
            {backendOnline ? 'Analyzer Engine Connected' : 'Engine Offline'}
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <div className="dashboard-grid">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="glass-panel">
            <div className="history-title">
              <span>Analysis Logs</span>
              {history.length > 0 && (
                <button className="history-clear-btn" onClick={clearHistory}>
                  Clear Cache
                </button>
              )}
            </div>

            <div className="history-list">
              {history.length === 0 ? (
                <div className="history-empty">
                  <Clock size={24} style={{ marginBottom: '0.5rem', opacity: 0.5 }} />
                  <p>No recent analysis logs found</p>
                </div>
              ) : (
                history.map((item, idx) => (
                  <div 
                    key={item.file_hash + idx} 
                    className="history-item"
                    onClick={() => loadHistoryItem(item)}
                  >
                    <div className="history-item-details">
                      <span className="history-item-filename">{item.filename}</span>
                      <span className="history-item-meta">
                        {item.is_image ? 'Image' : 'Video'} • {formatTime(item.timestamp)}
                      </span>
                    </div>
                    <div className={`history-item-score ${item.is_fake ? 'danger' : 'safe'}`}>
                      {formatPercentage(item.global_fake_score)}%
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="glass-panel" style={{ background: 'linear-gradient(180deg, rgba(139, 92, 246, 0.05) 0%, rgba(13, 20, 38, 0.65) 100%)' }}>
            <h3 style={{ fontSize: '0.9rem', marginBottom: '0.75rem', color: 'var(--secondary)', fontWeight: 600 }}>System Specifications</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Backend Architecture:</span>
                <span style={{ color: '#fff', fontWeight: 500 }}>FastAPI (Uvicorn)</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Inference Processor:</span>
                <span style={{ color: '#fff', fontWeight: 500 }}>PyTorch CPU/CUDA</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Detection Backbones:</span>
                <span style={{ color: '#fff', fontWeight: 500 }}>ViT & OpenCV Haar</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>State Database:</span>
                <span style={{ color: '#fff', fontWeight: 500 }}>Redis Cache</span>
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="main-panel">
          
          {/* 1. Upload & Initial Mode */}
          {status === null && (
            <div 
              className="glass-panel upload-zone"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={triggerFileSelect}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                style={{ display: 'none' }}
                accept="image/*,video/*"
              />
              <div className="upload-icon">
                <UploadCloud size={32} />
              </div>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 600 }}>Submit Media for Verification</h2>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', maxWidth: '360px', margin: '0 auto' }}>
                Drag and drop your video or image file here, or click to browse local files. Supports MP4, AVI, MOV, JPG, PNG.
              </p>
            </div>
          )}

          {/* 2. Loading / Processing Mode */}
          {(status === 'uploading' || status === 'processing') && (
            <div className="glass-panel progress-panel">
              <div className="progress-header">
                <div className="progress-stage">{stage}</div>
                <div className="progress-filename">{file?.name}</div>
              </div>
              
              <div className="progress-bar-container">
                <div className="progress-bar" style={{ width: `${progress}%` }}></div>
              </div>

              <div className="progress-percentage">
                {progress}%
              </div>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                Running facial localization and transformer model inference...
              </p>
            </div>
          )}

          {/* 3. Error / Failure Mode */}
          {status === 'failed' && (
            <div className="glass-panel" style={{ borderColor: 'var(--danger)', padding: '2rem', textAlign: 'center' }}>
              <AlertCircle size={48} color="var(--danger)" style={{ margin: '0 auto 1rem' }} />
              <h3 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '0.5rem' }}>Analysis Failed</h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '1.5rem' }}>
                {error || "An unknown error occurred while analyzing the media."}
              </p>
              <button className="btn" onClick={() => setStatus(null)}>
                Try Again
              </button>
            </div>
          )}

          {/* 4. Complete / Results Mode */}
          {status === 'completed' && result && (
            <>
              {/* Top Row: Gauge & Overview */}
              <div className="results-summary-row">
                {/* Gauge Card */}
                <div className="glass-panel gauge-card">
                  <svg width="180" height="180" className="gauge-svg">
                    <circle cx="90" cy="90" r="75" className="gauge-track" />
                    <circle 
                      cx="90" 
                      cy="90" 
                      r="75" 
                      className={`gauge-fill ${result.is_fake ? 'danger' : 'safe'}`}
                      strokeDasharray={2 * Math.PI * 75}
                      strokeDashoffset={(2 * Math.PI * 75) * (1 - result.global_fake_score)}
                    />
                  </svg>
                  
                  <div className="gauge-center-text">
                    <span className="gauge-score">
                      {formatPercentage(result.global_fake_score)}
                      <span className="gauge-pct-sign">%</span>
                    </span>
                    <span className="gauge-label">Fake Score</span>
                  </div>

                  <div className={`verdict-tag ${result.is_fake ? 'danger' : 'safe'}`}>
                    {result.is_fake ? 'Deepfake Flagged' : 'Verified Authentic'}
                  </div>
                </div>

                {/* Details Statistics Card */}
                <div className="glass-panel stats-card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                    <div>
                      <h2 style={{ fontSize: '1.25rem', fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '320px', whiteSpace: 'nowrap' }}>
                        {result.filename}
                      </h2>
                      <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.35rem', marginTop: '0.25rem' }}>
                        <Cpu size={14} /> 
                        {result.used_vit_model 
                          ? 'Transformer Model Active (ViT_Deepfake_Detection)' 
                          : 'Local Heuristics & Image Forensics Active'}
                      </p>
                    </div>
                    
                    <button className="btn btn-secondary" onClick={() => setStatus(null)} style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}>
                      <RefreshCw size={12} /> Reset
                    </button>
                  </div>

                  <div className="stats-grid">
                    <div className="stat-item">
                      <div className="stat-label">Analysis Target</div>
                      <div className="stat-value" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        {result.is_image ? <ImageIcon size={18} color="var(--secondary)" /> : <Video size={18} color="var(--secondary)" />}
                        {result.is_image ? 'Image File' : 'Video File'}
                      </div>
                    </div>
                    <div className="stat-item">
                      <div className="stat-label">Processing Latency</div>
                      <div className="stat-value">{result.processing_time_sec}s</div>
                    </div>
                    <div className="stat-item">
                      <div className="stat-label">Sampled Frames</div>
                      <div className="stat-value">{result.total_frames_analyzed}</div>
                    </div>
                    <div className="stat-item">
                      <div className="stat-label">Total Faces Detected</div>
                      <div className="stat-value">{result.total_faces_detected}</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Middle Row: Forensics Heuristics & Secondary Signatures */}
              <div className="glass-panel heuristics-card">
                <div className="heuristics-title">
                  <Layers size={18} color="var(--primary)" />
                  <span>Pixel-Level Visual Artifact Markers</span>
                </div>
                
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem' }}>
                  {/* Heuristic 1 */}
                  <div className="heuristic-row">
                    <div className="heuristic-info">
                      <span className="heuristic-label">Boundary Blurring</span>
                      <span className="heuristic-val" style={{ color: getHeuristicColor(result.average_heuristics.blur_artifact_score) }}>
                        {formatPercentage(result.average_heuristics.blur_artifact_score)}%
                      </span>
                    </div>
                    <div className="heuristic-bar-bg">
                      <div 
                        className="heuristic-bar-fill" 
                        style={{ 
                          width: `${result.average_heuristics.blur_artifact_score * 100}%`,
                          backgroundColor: getHeuristicColor(result.average_heuristics.blur_artifact_score)
                        }}
                      ></div>
                    </div>
                  </div>
                  
                  {/* Heuristic 2 */}
                  <div className="heuristic-row">
                    <div className="heuristic-info">
                      <span className="heuristic-label">Spectral Frequency Anomalies</span>
                      <span className="heuristic-val" style={{ color: getHeuristicColor(result.average_heuristics.frequency_anomaly_score) }}>
                        {formatPercentage(result.average_heuristics.frequency_anomaly_score)}%
                      </span>
                    </div>
                    <div className="heuristic-bar-bg">
                      <div 
                        className="heuristic-bar-fill" 
                        style={{ 
                          width: `${result.average_heuristics.frequency_anomaly_score * 100}%`,
                          backgroundColor: getHeuristicColor(result.average_heuristics.frequency_anomaly_score)
                        }}
                      ></div>
                    </div>
                  </div>
                  
                  {/* Heuristic 3 */}
                  <div className="heuristic-row">
                    <div className="heuristic-info">
                      <span className="heuristic-label">Lighting & Skin Tone Outliers</span>
                      <span className="heuristic-val" style={{ color: getHeuristicColor(result.average_heuristics.color_anomaly_score) }}>
                        {formatPercentage(result.average_heuristics.color_anomaly_score)}%
                      </span>
                    </div>
                    <div className="heuristic-bar-bg">
                      <div 
                        className="heuristic-bar-fill" 
                        style={{ 
                          width: `${result.average_heuristics.color_anomaly_score * 100}%`,
                          backgroundColor: getHeuristicColor(result.average_heuristics.color_anomaly_score)
                        }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Bottom Section 1: Temporal Analysis Chart (Only for videos) */}
              {!result.is_image && result.frames && result.frames.length > 1 && (
                <div className="glass-panel">
                  <div className="heuristics-title">
                    <TrendingUp size={18} color="var(--secondary)" />
                    <span>Temporal Score Analysis (Select points to view frame details)</span>
                  </div>
                  
                  {/* Custom SVG Line Chart */}
                  <div className="chart-container">
                    <svg style={{ width: '100%', height: '100%', overflow: 'visible' }}>
                      {/* Grid Lines */}
                      <line x1="0" y1="0" x2="100%" y2="0" stroke="rgba(255,255,255,0.05)" strokeDasharray="3" />
                      <line x1="0" y1="60" x2="100%" y2="60" stroke="rgba(255,255,255,0.05)" strokeDasharray="3" />
                      <line x1="0" y1="120" x2="100%" y2="120" stroke="rgba(255,255,255,0.05)" strokeDasharray="3" />
                      
                      {/* Render line connecting frames */}
                      <path
                        d={(() => {
                          const w = 700; // arbitrary reference width
                          const points = result.frames.map((f, i) => {
                            const xPct = (i / (result.frames.length - 1)) * 100;
                            // Find highest face score on this frame, or 0 if no face
                            const frameScore = f.faces.length > 0 
                              ? Math.max(...f.faces.map(face => face.fake_score)) 
                              : 0.0;
                            const yVal = 120 - (frameScore * 120);
                            return `${xPct}%,${yVal}`;
                          });
                          return `M ${points.map((p, i) => `${i === 0 ? '' : 'L'} ${p}`).join(' ')}`;
                        })()}
                        fill="none"
                        stroke="url(#chart-gradient)"
                        strokeWidth="3"
                        style={{ vectorEffect: 'non-scaling-stroke' }}
                      />

                      {/* Gradients definitions */}
                      <defs>
                        <linearGradient id="chart-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                          <stop offset="0%" stopColor="var(--primary)" />
                          <stop offset="100%" stopColor="var(--secondary)" />
                        </linearGradient>
                      </defs>

                      {/* Interactive nodes */}
                      {result.frames.map((f, i) => {
                        const xPct = `${(i / (result.frames.length - 1)) * 100}%`;
                        const frameScore = f.faces.length > 0 
                          ? Math.max(...f.faces.map(face => face.fake_score)) 
                          : 0.0;
                        const yVal = 120 - (frameScore * 120);
                        const isSelected = selectedFrameIdx === f.frame_idx;

                        return (
                          <g key={f.frame_idx} style={{ cursor: 'pointer' }} onClick={() => setSelectedFrameIdx(f.frame_idx)}>
                            <circle 
                              cx={xPct} 
                              cy={yVal} 
                              r={isSelected ? "7" : "5"} 
                              fill={isSelected ? "var(--secondary)" : "var(--primary)"}
                              stroke={isSelected ? "#fff" : "rgba(255, 255, 255, 0.2)"}
                              strokeWidth="2"
                              style={{ transition: 'all 0.2s' }}
                            />
                            {/* Hidden larger circle for easy hover target */}
                            <circle 
                              cx={xPct} 
                              cy={yVal} 
                              r="15" 
                              fill="transparent"
                            />
                          </g>
                        );
                      })}
                    </svg>
                  </div>
                  
                  {/* Frame Detail Inspector */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '1.5rem', marginTop: '1.5rem' }}>
                    <div style={{ background: 'rgba(255, 255, 255, 0.01)', border: '1px solid rgba(255, 255, 255, 0.03)', borderRadius: '12px', padding: '1rem' }}>
                      <h4 style={{ fontSize: '0.9rem', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
                        <Camera size={14} color="var(--primary)" /> Selected Frame Inspector (Frame #{currentFrameData?.frame_idx + 1})
                      </h4>
                      <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                        Timestamp: <strong style={{ color: '#fff' }}>{currentFrameData?.timestamp}s</strong> • 
                        Detected Faces: <strong style={{ color: '#fff' }}>{currentFrameData?.faces?.length || 0}</strong>
                      </p>
                      
                      {currentFrameData?.faces?.length > 0 ? (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '1rem' }}>
                          {currentFrameData.faces.map((face) => (
                            <div 
                              key={face.face_id}
                              style={{ 
                                display: 'flex', 
                                alignItems: 'center', 
                                gap: '1rem', 
                                background: 'rgba(255, 255, 255, 0.02)',
                                border: '1px solid rgba(255,255,255,0.04)',
                                borderRadius: '8px', 
                                padding: '0.5rem 0.75rem' 
                              }}
                            >
                              <img 
                                src={`data:image/jpeg;base64,${face.crop_b64}`} 
                                alt="Face Crop" 
                                style={{ width: '45px', height: '45px', borderRadius: '6px', objectFit: 'cover', border: '1px solid rgba(255,255,255,0.1)' }} 
                              />
                              <div style={{ flex: 1 }}>
                                <div style={{ fontSize: '0.8rem', fontWeight: 600 }}>Face #{face.face_id + 1}</div>
                                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Bounding Box: [{face.box.join(', ')}]</div>
                              </div>
                              <div style={{ textAlign: 'right' }}>
                                <div className={`face-card-score ${face.fake_score > 0.5 ? 'danger' : 'safe'}`}>
                                  {formatPercentage(face.fake_score)}% Fake
                                </div>
                                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                                  Confidence: {formatPercentage(face.confidence)}%
                                </div>
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

                    <div style={{ background: 'rgba(255, 255, 255, 0.01)', border: '1px solid rgba(255, 255, 255, 0.03)', borderRadius: '12px', padding: '1rem' }}>
                      <h4 style={{ fontSize: '0.9rem', marginBottom: '0.75rem', fontWeight: 600 }}>Frame Spectral Metrics</h4>
                      {currentFrameData?.faces?.length > 0 ? (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.03)', paddingBottom: '0.25rem' }}>
                            <span>Sharpness Val:</span>
                            <span style={{ color: '#fff', fontWeight: 500 }}>
                              {currentFrameData.faces[0].heuristics.sharpness_val}
                            </span>
                          </div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.03)', paddingBottom: '0.25rem' }}>
                            <span>Freq Ratio:</span>
                            <span style={{ color: '#fff', fontWeight: 500 }}>
                              {currentFrameData.faces[0].heuristics.freq_ratio}
                            </span>
                          </div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.03)', paddingBottom: '0.25rem' }}>
                            <span>Color Depth Val:</span>
                            <span style={{ color: '#fff', fontWeight: 500 }}>
                              {currentFrameData.faces[0].heuristics.color_depth_val}
                            </span>
                          </div>
                          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span>Artifact Sum:</span>
                            <span style={{ 
                              color: getHeuristicColor(
                                (currentFrameData.faces[0].heuristics.blur_artifact_score + 
                                 currentFrameData.faces[0].heuristics.frequency_anomaly_score) / 2
                              ), 
                              fontWeight: 600 
                            }}>
                              {formatPercentage((currentFrameData.faces[0].heuristics.blur_artifact_score + currentFrameData.faces[0].heuristics.frequency_anomaly_score) / 2)}%
                            </span>
                          </div>
                        </div>
                      ) : (
                        <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem', fontSize: '0.8rem' }}>
                          No frame metrics available.
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Bottom Section 2: Detected Faces Gallery */}
              {allFaces.length > 0 && (
                <div className="glass-panel">
                  <div className="faces-title">
                    <Camera size={18} color="var(--primary)" />
                    <span>Detected Face Profiles Library ({allFaces.length} total)</span>
                  </div>
                  
                  <div className="faces-grid">
                    {allFaces.map((face, index) => (
                      <div 
                        key={index} 
                        className="face-card"
                        onClick={() => {
                          if (!result.is_image) {
                            setSelectedFrameIdx(face.frame_idx);
                          }
                        }}
                      >
                        <img 
                          src={`data:image/jpeg;base64,${face.crop_b64}`} 
                          className="face-crop-img" 
                          alt="Face Crop" 
                        />
                        <div className="face-info-overlay">
                          <div className={`face-card-score ${face.fake_score > 0.5 ? 'danger' : 'safe'}`}>
                            {formatPercentage(face.fake_score)}% Fake
                          </div>
                          <div className="face-card-meta">
                            {!result.is_image ? `@ ${face.timestamp}s` : 'Image Target'}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

        </main>
      </div>
    </div>
  );
}

export default App;
