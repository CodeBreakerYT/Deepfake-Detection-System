import { TrendingUp } from 'lucide-react';

/**
 * Generic temporal score chart. `points` is an array of { idx, score, label }.
 * `selectedIdx` / `onSelect` drive which point is highlighted.
 */
export default function Timeline({ title, points, selectedIdx, onSelect }) {
  if (!points || points.length < 2) return null;

  const path = `M ${points
    .map((p, i) => {
      const xPct = (i / (points.length - 1)) * 100;
      const yVal = 120 - p.score * 120;
      return `${i === 0 ? '' : 'L'} ${xPct}%,${yVal}`;
    })
    .join(' ')}`;

  return (
    <div className="glass-panel">
      <div className="heuristics-title">
        <TrendingUp size={18} color="var(--secondary)" />
        <span>{title}</span>
      </div>

      <div className="chart-container">
        <svg style={{ width: '100%', height: '100%', overflow: 'visible' }}>
          <line x1="0" y1="0" x2="100%" y2="0" stroke="rgba(255,255,255,0.05)" strokeDasharray="3" />
          <line x1="0" y1="60" x2="100%" y2="60" stroke="rgba(255,255,255,0.05)" strokeDasharray="3" />
          <line x1="0" y1="120" x2="100%" y2="120" stroke="rgba(255,255,255,0.05)" strokeDasharray="3" />

          <path
            d={path}
            fill="none"
            stroke="url(#chart-gradient)"
            strokeWidth="3"
            style={{ vectorEffect: 'non-scaling-stroke' }}
          />

          <defs>
            <linearGradient id="chart-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="var(--primary)" />
              <stop offset="100%" stopColor="var(--secondary)" />
            </linearGradient>
          </defs>

          {points.map((p, i) => {
            const xPct = `${(i / (points.length - 1)) * 100}%`;
            const yVal = 120 - p.score * 120;
            const isSelected = selectedIdx === p.idx;

            return (
              <g key={p.idx} style={{ cursor: 'pointer' }} onClick={() => onSelect(p.idx)}>
                <circle
                  cx={xPct}
                  cy={yVal}
                  r={isSelected ? '7' : '5'}
                  fill={isSelected ? 'var(--secondary)' : 'var(--primary)'}
                  stroke={isSelected ? '#fff' : 'rgba(255, 255, 255, 0.2)'}
                  strokeWidth="2"
                  style={{ transition: 'all 0.2s' }}
                />
                <circle cx={xPct} cy={yVal} r="15" fill="transparent" />
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
