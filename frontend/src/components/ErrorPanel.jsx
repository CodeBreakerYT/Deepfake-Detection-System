import { AlertCircle } from 'lucide-react';

export default function ErrorPanel({ message, onRetry }) {
  return (
    <div className="glass-panel" style={{ borderColor: 'var(--danger)', padding: '2rem', textAlign: 'center' }}>
      <AlertCircle size={48} color="var(--danger)" style={{ margin: '0 auto 1rem' }} />
      <h3 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '0.5rem' }}>Analysis Failed</h3>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '1.5rem' }}>
        {message || 'An unknown error occurred while analyzing the media.'}
      </p>
      <button className="btn" onClick={onRetry}>Try Again</button>
    </div>
  );
}
