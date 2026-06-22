import { Layers } from 'lucide-react';
import { formatPct, getHeuristicColor } from '../utils';

export default function HeuristicBars({ title, items }) {
  return (
    <div className="glass-panel heuristics-card">
      <div className="heuristics-title">
        <Layers size={18} color="var(--primary)" />
        <span>{title}</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${items.length}, 1fr)`, gap: '1.5rem' }}>
        {items.map((item) => (
          <div className="heuristic-row" key={item.label}>
            <div className="heuristic-info">
              <span className="heuristic-label">{item.label}</span>
              <span className="heuristic-val" style={{ color: getHeuristicColor(item.value) }}>
                {formatPct(item.value)}%
              </span>
            </div>
            <div className="heuristic-bar-bg">
              <div
                className="heuristic-bar-fill"
                style={{ width: `${item.value * 100}%`, backgroundColor: getHeuristicColor(item.value) }}
              ></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
