import { useRef } from 'react';
import { UploadCloud } from 'lucide-react';

export default function UploadZone({ accept, title, subtitle, onFile }) {
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => e.preventDefault();
  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFile(e.dataTransfer.files[0]);
    }
  };
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onFile(e.target.files[0]);
    }
  };

  return (
    <div
      className="glass-panel upload-zone"
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current.click()}
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        style={{ display: 'none' }}
        accept={accept}
      />
      <div className="upload-icon">
        <UploadCloud size={32} />
      </div>
      <h2 style={{ fontSize: '1.25rem', fontWeight: 600 }}>{title}</h2>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', maxWidth: '380px', margin: '0 auto' }}>
        {subtitle}
      </p>
    </div>
  );
}
