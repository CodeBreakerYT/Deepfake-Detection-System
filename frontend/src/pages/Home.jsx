import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Image as ImageIcon, Video, Mic, ShieldCheck, Clock, ArrowRight } from 'lucide-react';
import { fetchHistory } from '../api';

const CARDS = [
  {
    to: '/image',
    icon: ImageIcon,
    title: 'Image Analysis',
    desc: 'Detect face-swaps, GAN artifacts, and pixel-level forensic anomalies in photos.',
  },
  {
    to: '/video',
    icon: Video,
    title: 'Video Analysis',
    desc: 'Sample frames, track faces over time, and flag temporal inconsistencies.',
  },
  {
    to: '/voices',
    icon: Mic,
    title: 'Voice Analysis',
    desc: 'Catch AI-cloned speech and synthetic vocoder artifacts in audio recordings.',
  },
];

export default function Home() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    fetchHistory().then(setHistory).catch(() => setHistory([]));
  }, []);

  const total = history.length;
  const flagged = history.filter((h) => h.is_fake).length;

  return (
    <main className="main-panel">
      <div className="hero-panel glass-panel">
        <div className="hero-badge">
          <ShieldCheck size={14} /> No account required
        </div>
        <h1 className="hero-title">Verify media authenticity in seconds</h1>
        <p className="hero-subtitle">
          DeepShield combines transformer deep learning models with classical pixel and acoustic forensics
          to detect deepfakes across images, video, and voice — no login, no friction.
        </p>
        <div className="hero-stats">
          <div>
            <span className="hero-stat-value">{total}</span>
            <span className="hero-stat-label">Analyses Run</span>
          </div>
          <div>
            <span className="hero-stat-value">{flagged}</span>
            <span className="hero-stat-label">Flagged as Fake</span>
          </div>
        </div>
      </div>

      <div className="home-cards-grid">
        {CARDS.map(({ to, icon: Icon, title, desc }) => (
          <Link to={to} className="glass-panel home-card" key={to}>
            <div className="home-card-icon">
              <Icon size={24} />
            </div>
            <h3>{title}</h3>
            <p>{desc}</p>
            <span className="home-card-cta">
              Start Analyzing <ArrowRight size={14} />
            </span>
          </Link>
        ))}
      </div>

      {total > 0 && (
        <div className="glass-panel">
          <div className="history-title">
            <span>Recent Activity</span>
          </div>
          <div className="history-list">
            {history.slice(0, 6).map((item, idx) => (
              <div className="history-item" key={item.file_hash + idx}>
                <div className="history-item-details">
                  <span className="history-item-filename">{item.filename}</span>
                  <span className="history-item-meta">
                    <Clock size={11} style={{ display: 'inline', marginRight: '4px' }} />
                    {item.is_audio ? 'Voice' : item.is_image ? 'Image' : 'Video'}
                  </span>
                </div>
                <div className={`history-item-score ${item.is_fake ? 'danger' : 'safe'}`}>
                  {Math.round(item.global_fake_score * 100)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </main>
  );
}
