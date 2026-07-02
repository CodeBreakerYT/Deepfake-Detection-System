import { Mic, Cpu, AudioLines } from 'lucide-react';
import UploadZone from '../components/UploadZone';
import ProgressPanel from '../components/ProgressPanel';
import ErrorPanel from '../components/ErrorPanel';
import ResultGauge from '../components/ResultGauge';
import { formatPct } from '../utils';
import HeuristicBars from '../components/HeuristicBars';
import Timeline from '../components/Timeline';
import { useDetection } from '../hooks/useDetection';

export default function VoicePage() {
  const { file, status, progress, stage, result, error, selectedIdx, setSelectedIdx, processUpload, reset } =
    useDetection('/api/detect/audio');

  const currentSegment = result?.segments?.find((s) => s.segment_idx === selectedIdx) || result?.segments?.[0];

  const timelinePoints = result?.segments?.map((s) => ({ idx: s.segment_idx, score: s.fake_score }));

  return (
    <main className="main-panel">
      <div className="page-intro">
        <h1>Voice Deepfake Analysis</h1>
        <p>Upload a voice recording to detect AI-generated speech, voice cloning, and synthetic vocoder artifacts.</p>
      </div>

      {status === null && (
        <UploadZone
          accept="audio/*"
          title="Submit Audio for Verification"
          subtitle="Drag and drop a WAV, MP3, FLAC, OGG, or M4A file here, or click to browse."
          onFile={processUpload}
        />
      )}

      {(status === 'uploading' || status === 'processing') && (
        <ProgressPanel
          filename={file?.name}
          stage={stage}
          progress={progress}
          hint="Segmenting audio and running acoustic forensic + voice model inference..."
        />
      )}

      {status === 'failed' && <ErrorPanel message={error} onRetry={reset} />}

      {status === 'completed' && result && (
        <>
          <div className="glass-panel" style={{ padding: '1rem', marginBottom: '2rem', textAlign: 'center' }}>
            <h3 style={{ fontSize: '1.1rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
              <Mic size={18} color="var(--primary)" /> Audio Playback & Analysis
            </h3>
            {file && (
              <audio 
                src={URL.createObjectURL(file)} 
                controls 
                style={{ width: '100%', maxWidth: '600px', marginBottom: '1rem' }} 
              />
            )}
            <div style={{ textAlign: 'left', background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', fontSize: '0.9rem' }}>
              <h4 style={{ marginBottom: '0.5rem', color: 'var(--primary)' }}>Transcription</h4>
              <p style={{ color: 'var(--text-secondary)', fontStyle: 'italic', marginBottom: '1rem' }}>
                "{result.transcript || 'No speech detected or transcription unavailable.'}"
              </p>
              <h4 style={{ marginBottom: '0.5rem', color: '#ef4444' }}>Deepfake Timestamps (Fake Audio Detected)</h4>
              <ul style={{ listStyle: 'none', padding: 0, margin: 0, color: 'var(--text-secondary)' }}>
                {result.segments?.filter(s => s.is_fake).length > 0 ? (
                  result.segments.filter(s => s.is_fake).map((s, idx) => (
                    <li key={idx} style={{ marginBottom: '0.25rem' }}>
                      <span style={{ color: '#ef4444', fontWeight: 'bold' }}>{s.start_time}s - {s.end_time}s</span> (Score: {formatPct(s.fake_score)}%)
                    </li>
                  ))
                ) : (
                  <li>No synthetic or cloned audio segments detected.</li>
                )}
              </ul>
            </div>
          </div>

          <ResultGauge
            result={{
              ...result,
              stats: [
                { label: 'Analysis Target', value: <><Mic size={18} color="var(--secondary)" /> Audio File</> },
                { label: 'Processing Latency', value: `${result.processing_time_sec}s` },
                { label: 'Clip Duration', value: `${result.duration_sec}s` },
                { label: 'Segments Analyzed', value: result.total_segments_analyzed },
              ],
            }}
            icon={<Cpu size={14} />}
            modeLine={result.used_voice_model ? 'Transformer Model Active (wav2vec2-deepfake-voice-detector)' : 'Local Acoustic Heuristics Active'}
            onReset={reset}
          />

          <HeuristicBars
            title="Acoustic Artifact Markers"
            items={[
              { label: 'Pitch Jitter Anomaly', value: result.average_heuristics.pitch_jitter_score },
              { label: 'Spectral Flatness Anomaly', value: result.average_heuristics.spectral_flatness_score },
              { label: 'Energy/Silence Anomaly', value: result.average_heuristics.silence_anomaly_score },
            ]}
          />

          {timelinePoints && timelinePoints.length > 1 && (
            <>
              <Timeline
                title="Per-Segment Score Timeline (Select points to inspect)"
                points={timelinePoints}
                selectedIdx={selectedIdx}
                onSelect={setSelectedIdx}
              />

              <div className="glass-panel">
                <h4 style={{ fontSize: '0.9rem', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
                  <AudioLines size={14} color="var(--primary)" /> Selected Segment Inspector (Segment #{(currentSegment?.segment_idx ?? 0) + 1})
                </h4>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                  Window: <strong style={{ color: '#fff' }}>{currentSegment?.start_time}s – {currentSegment?.end_time}s</strong> •
                  Fake Score: <strong className={currentSegment?.is_fake ? 'face-card-score danger' : 'face-card-score safe'}>{formatPct(currentSegment?.fake_score || 0)}%</strong>
                </p>

                {currentSegment && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.03)', paddingBottom: '0.25rem' }}>
                      <span>Pitch Jitter Value:</span>
                      <span style={{ color: '#fff', fontWeight: 500 }}>{currentSegment.heuristics.pitch_jitter_val}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.03)', paddingBottom: '0.25rem' }}>
                      <span>Spectral Flatness Value:</span>
                      <span style={{ color: '#fff', fontWeight: 500 }}>{currentSegment.heuristics.spectral_flatness_val}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Energy Variation (CV):</span>
                      <span style={{ color: '#fff', fontWeight: 500 }}>{currentSegment.heuristics.energy_variation_val}</span>
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </>
      )}
    </main>
  );
}
