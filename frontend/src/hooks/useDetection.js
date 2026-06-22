import { useState, useRef } from 'react';
import { uploadAndTrack } from '../api';

/**
 * Shared upload -> progress -> result/error state machine used by the
 * Image, Video, and Voices pages against their respective detect endpoints.
 */
export function useDetection(endpoint) {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState(null); // 'uploading' | 'processing' | 'completed' | 'failed' | null
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const startedAt = useRef(0);

  const processUpload = (selectedFile) => {
    setFile(selectedFile);
    setStatus('uploading');
    setProgress(5);
    setStage('Uploading to analysis server...');
    setError(null);
    startedAt.current = Date.now();

    uploadAndTrack(endpoint, selectedFile, {
      onStage: ({ status: s, progress: p, stage: st }) => {
        setStatus(s);
        setProgress(p);
        setStage(st);
      },
      onResult: (res) => {
        setResult(res);
        setStatus('completed');
        setProgress(100);
        setSelectedIdx(0);
      },
      onError: (message) => {
        setError(message);
        setStatus('failed');
      },
    });
  };

  const reset = () => {
    setFile(null);
    setStatus(null);
    setProgress(0);
    setStage('');
    setResult(null);
    setError(null);
    setSelectedIdx(0);
  };

  const loadFromHistory = (item) => {
    setResult(item);
    setFile({ name: item.filename });
    setStatus('completed');
    setSelectedIdx(0);
  };

  return {
    file, status, progress, stage, result, error, selectedIdx,
    setSelectedIdx, processUpload, reset, loadFromHistory,
  };
}
