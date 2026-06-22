import { RefreshCw } from 'lucide-react';
import { formatPct } from '../utils';

export default function ResultGauge({ result, icon, modeLine, onReset }) {
  return (
    <div className="results-summary-row">
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
            {formatPct(result.global_fake_score)}
            <span className="gauge-pct-sign">%</span>
          </span>
          <span className="gauge-label">Fake Score</span>
        </div>

        <div className={`verdict-tag ${result.is_fake ? 'danger' : 'safe'}`}>
          {result.is_fake ? 'Deepfake Flagged' : 'Verified Authentic'}
        </div>
      </div>

      <div className="glass-panel stats-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
          <div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '320px', whiteSpace: 'nowrap' }}>
              {result.filename}
            </h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.35rem', marginTop: '0.25rem' }}>
              {icon} {modeLine}
            </p>
          </div>

          <button className="btn btn-secondary" onClick={onReset} style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}>
            <RefreshCw size={12} /> Reset
          </button>
        </div>

        <div className="stats-grid">
          {result.stats.map((s) => (
            <div className="stat-item" key={s.label}>
              <div className="stat-label">{s.label}</div>
              <div className="stat-value" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                {s.value}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
