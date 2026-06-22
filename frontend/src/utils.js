export const formatPct = (val) => `${Math.round(val * 100)}`;

export const getHeuristicColor = (score) => {
  if (score > 0.6) return 'var(--danger)';
  if (score > 0.3) return 'var(--warning)';
  return 'var(--success)';
};
