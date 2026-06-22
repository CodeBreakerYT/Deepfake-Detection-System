export default function ProgressPanel({ filename, stage, progress, hint }) {
  return (
    <div className="glass-panel progress-panel">
      <div className="progress-header">
        <div className="progress-stage">{stage}</div>
        <div className="progress-filename">{filename}</div>
      </div>

      <div className="progress-bar-container">
        <div className="progress-bar" style={{ width: `${progress}%` }}></div>
      </div>

      <div className="progress-percentage">{progress}%</div>
      <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
        {hint}
      </p>
    </div>
  );
}
