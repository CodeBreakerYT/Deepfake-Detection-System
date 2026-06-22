import { NavLink } from 'react-router-dom';
import { ShieldAlert, Image as ImageIcon, Video, Mic, Info, Home as HomeIcon } from 'lucide-react';
import { useEffect, useState } from 'react';
import { checkBackendHealth } from '../api';

const TABS = [
  { to: '/', label: 'Home', icon: HomeIcon, end: true },
  { to: '/image', label: 'Image', icon: ImageIcon },
  { to: '/video', label: 'Video', icon: Video },
  { to: '/voices', label: 'Voices', icon: Mic },
  { to: '/about', label: 'About', icon: Info },
];

export default function Navbar() {
  const [backendOnline, setBackendOnline] = useState(false);

  useEffect(() => {
    const check = () => checkBackendHealth().then(setBackendOnline);
    check();
    const interval = setInterval(check, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
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

      <nav className="navbar-tabs">
        {TABS.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) => `navbar-tab ${isActive ? 'active' : ''}`}
          >
            <Icon size={16} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="system-status">
        <div className="status-badge">
          <span className={`status-indicator ${backendOnline ? 'online' : 'offline'}`}></span>
          {backendOnline ? 'Engine Connected' : 'Engine Offline'}
        </div>
      </div>
    </header>
  );
}
