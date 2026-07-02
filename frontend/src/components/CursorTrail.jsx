import { useEffect, useRef } from 'react';

const COLORS = ['#8b5cf6', '#06b6d4', '#f3f4f6', '#f59e0b', '#f43f5e'];

function randomBetween(min, max) {
  return Math.random() * (max - min) + min;
}

function drawStar(ctx, x, y, size, rotation, color, alpha) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rotation);
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  const spikes = 4;
  const outerRadius = size;
  const innerRadius = size * 0.35;
  for (let i = 0; i < spikes * 2; i++) {
    const r = i % 2 === 0 ? outerRadius : innerRadius;
    const angle = (Math.PI / spikes) * i;
    const px = Math.cos(angle) * r;
    const py = Math.sin(angle) * r;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = size * 2.5;
  ctx.fill();
  ctx.restore();
}

export default function CursorTrail() {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!window.matchMedia('(pointer: fine)').matches) return undefined;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let particles = [];
    let lastSpawn = 0;
    let rafId;

    function resize() {
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      canvas.style.width = `${window.innerWidth}px`;
      canvas.style.height = `${window.innerHeight}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    window.addEventListener('resize', resize);

    function spawn(x, y) {
      const now = performance.now();
      if (now - lastSpawn < 16) return;
      lastSpawn = now;

      for (let i = 0; i < 2; i++) {
        particles.push({
          x: x + randomBetween(-4, 4),
          y: y + randomBetween(-4, 4),
          vx: randomBetween(-0.5, 0.5),
          vy: randomBetween(-1, -0.2),
          size: randomBetween(2, 5),
          rotation: randomBetween(0, Math.PI * 2),
          rotationSpeed: randomBetween(-0.06, 0.06),
          color: COLORS[Math.floor(Math.random() * COLORS.length)],
          life: 1,
          decay: randomBetween(0.014, 0.024),
        });
      }
      if (particles.length > 240) {
        particles = particles.slice(particles.length - 240);
      }
    }

    function handleMove(e) {
      spawn(e.clientX, e.clientY);
    }
    function handleTouch(e) {
      if (e.touches && e.touches[0]) {
        spawn(e.touches[0].clientX, e.touches[0].clientY);
      }
    }
    window.addEventListener('mousemove', handleMove, { passive: true });
    window.addEventListener('touchmove', handleTouch, { passive: true });

    function tick() {
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.x += p.vx;
        p.y += p.vy;
        p.vy += 0.006;
        p.rotation += p.rotationSpeed;
        p.life -= p.decay;
        if (p.life <= 0) {
          particles.splice(i, 1);
          continue;
        }
        drawStar(ctx, p.x, p.y, p.size * p.life, p.rotation, p.color, Math.max(p.life, 0));
      }
      rafId = requestAnimationFrame(tick);
    }
    rafId = requestAnimationFrame(tick);

    return () => {
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('touchmove', handleTouch);
      cancelAnimationFrame(rafId);
    };
  }, []);

  return <canvas ref={canvasRef} className="cursor-trail-canvas" aria-hidden="true" />;
}
