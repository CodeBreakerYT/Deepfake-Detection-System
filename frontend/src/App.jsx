import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import CursorTrail from './components/CursorTrail';
import Home from './pages/Home';
import ImagePage from './pages/ImagePage';
import VideoPage from './pages/VideoPage';
import VoicePage from './pages/VoicePage';
import About from './pages/About';

function App() {
  return (
    <BrowserRouter>
      <div className="aurora-bg" aria-hidden="true">
        <span className="aurora-blob aurora-blob-1" />
        <span className="aurora-blob aurora-blob-2" />
        <span className="aurora-blob aurora-blob-3" />
      </div>
      <CursorTrail />
      <div className="container">
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/image" element={<ImagePage />} />
          <Route path="/video" element={<VideoPage />} />
          <Route path="/voices" element={<VoicePage />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
