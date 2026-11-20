/**
 * Entropy 스타일 인터랙티브 파티클 캔버스 (화이트/다크 모드 완벽 대응 + Blue/Orange 표준 팔레트)
 */
export function mountEntropyCanvas(canvas) {
  if (!canvas) return () => {};
  const ctx = canvas.getContext('2d');
  if (!ctx) return () => {};

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let width = 0;
  let height = 0;
  let dpr = Math.min(window.devicePixelRatio || 1, 2);
  let particles = [];
  let rafId = null;
  let frame = 0;
  const mouse = { x: -9999, y: -9999, active: false };

  function getColors() {
    const isLight = document.documentElement.dataset.theme === 'light';
    return isLight
      ? {
          fwdRgb: '37, 99, 235',
          backRgb: '234, 88, 12',
          mixRgb: '100, 116, 139',
          cursorLine: 'rgba(15, 23, 42, 0.28)',
          midLine: 'rgba(15, 23, 42, 0.14)',
        }
      : {
          fwdRgb: '59, 130, 246',
          backRgb: '249, 115, 22',
          mixRgb: '148, 163, 184',
          cursorLine: 'rgba(226, 232, 240, 0.3)',
          midLine: 'rgba(226, 232, 240, 0.14)',
        };
  }

  class Particle {
    constructor(x, y, order) {
      this.x = x;
      this.y = y;
      this.originalX = x;
      this.originalY = y;
      this.order = order;
      this.size = order ? 2.2 : 2.5;
      this.vx = (Math.random() - 0.5) * 1.5;
      this.vy = (Math.random() - 0.5) * 1.5;
      this.influence = 0;
      this.neighbors = [];
    }

    update() {
      const midX = width * 0.5;
      if (this.order) {
        const dx = this.originalX - this.x;
        const dy = this.originalY - this.y;
        let cx = 0;
        let cy = 0;
        for (let i = 0; i < this.neighbors.length; i++) {
          const nb = this.neighbors[i];
          if (!nb.order) {
            const dist = Math.hypot(this.x - nb.x, this.y - nb.y);
            const str = Math.max(0, 1 - dist / 130);
            cx += nb.vx * str;
            cy += nb.vy * str;
            if (str > this.influence) this.influence = str;
          }
        }
        this.x += dx * 0.055 * (1 - this.influence) + cx * this.influence;
        this.y += dy * 0.055 * (1 - this.influence) + cy * this.influence;
        this.influence *= 0.985;
      } else {
        this.vx += (Math.random() - 0.5) * 0.35;
        this.vy += (Math.random() - 0.5) * 0.35;
        this.vx *= 0.96;
        this.vy *= 0.96;
        this.x += this.vx;
        this.y += this.vy;

        if (this.x < midX || this.x > width) this.vx *= -1;
        if (this.y < 0 || this.y > height) this.vy *= -1;
        this.x = Math.max(midX, Math.min(width, this.x));
        this.y = Math.max(0, Math.min(height, this.y));
      }

      if (mouse.active) {
        const mdx = this.x - mouse.x;
        const mdy = this.y - mouse.y;
        const mdist = Math.hypot(mdx, mdy);
        if (mdist > 1 && mdist < 160) {
          const force = (1 - mdist / 160) * 1.8;
          this.x += (mdx / mdist) * force;
          this.y += (mdy / mdist) * force;
          this.influence = Math.min(1, this.influence + (1 - mdist / 160) * 0.35);
        }
      }
    }

    draw(colors) {
      const mdist = mouse.active ? Math.hypot(this.x - mouse.x, this.y - mouse.y) : 999;
      const hoverBoost = mdist < 170 ? 1 - mdist / 170 : 0;
      const alpha = Math.min(1, 0.45 + hoverBoost * 0.5);
      const rgb = this.order ? colors.fwdRgb : colors.backRgb;
      ctx.fillStyle = `rgba(${rgb}, ${alpha.toFixed(2)})`;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size + hoverBoost * 2.0, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function buildParticles() {
    particles = [];
    const cols = Math.max(14, Math.min(24, Math.round(width / 64)));
    const rows = Math.max(8, Math.min(14, Math.round(height / 64)));
    const stepX = width / cols;
    const stepY = height / rows;

    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < rows; j++) {
        const x = stepX * i + stepX * 0.5;
        const y = stepY * j + stepY * 0.5;
        particles.push(new Particle(x, y, x < width * 0.5));
      }
    }
    updateNeighbors();
  }

  function updateNeighbors() {
    const threshold = 115;
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      p.neighbors = [];
      for (let j = 0; j < particles.length; j++) {
        if (i === j) continue;
        const o = particles[j];
        if (Math.abs(p.x - o.x) > threshold || Math.abs(p.y - o.y) > threshold) continue;
        if (Math.hypot(p.x - o.x, p.y - o.y) < threshold) {
          p.neighbors.push(o);
        }
      }
    }
  }

  function resize() {
    const rect = canvas.parentElement.getBoundingClientRect();
    width = Math.max(320, rect.width);
    height = Math.max(360, rect.height);
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    buildParticles();
  }

  function renderFrame() {
    ctx.clearRect(0, 0, width, height);
    const colors = getColors();

    if (frame % 35 === 0) updateNeighbors();

    const gradLine = ctx.createLinearGradient(0, 0, 0, height);
    gradLine.addColorStop(0, 'rgba(0,0,0,0)');
    gradLine.addColorStop(0.5, colors.midLine);
    gradLine.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.strokeStyle = gradLine;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(width * 0.5, 0);
    ctx.lineTo(width * 0.5, height);
    ctx.stroke();

    if (mouse.active) {
      const glow = ctx.createRadialGradient(mouse.x, mouse.y, 0, mouse.x, mouse.y, 180);
      glow.addColorStop(0, `rgba(${colors.fwdRgb}, 0.12)`);
      glow.addColorStop(0.6, `rgba(${colors.backRgb}, 0.04)`);
      glow.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, width, height);
    }

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      if (!reducedMotion) p.update();

      for (let j = 0; j < p.neighbors.length; j++) {
        const nb = p.neighbors[j];
        const dist = Math.hypot(p.x - nb.x, p.y - nb.y);
        if (dist < 76) {
          const alpha = 0.2 * (1 - dist / 76);
          const rgb = p.order && nb.order ? colors.fwdRgb : !p.order && !nb.order ? colors.backRgb : colors.mixRgb;
          ctx.strokeStyle = `rgba(${rgb}, ${alpha.toFixed(3)})`;
          ctx.lineWidth = 0.9;
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(nb.x, nb.y);
          ctx.stroke();
        }
      }

      if (mouse.active) {
        const md = Math.hypot(p.x - mouse.x, p.y - mouse.y);
        if (md < 135) {
          ctx.strokeStyle = colors.cursorLine;
          ctx.lineWidth = 0.9;
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(mouse.x, mouse.y);
          ctx.stroke();
        }
      }

      p.draw(colors);
    }

    frame++;
    if (!reducedMotion) rafId = requestAnimationFrame(renderFrame);
  }

  function onPointerMove(e) {
    const rect = canvas.getBoundingClientRect();
    mouse.x = e.clientX - rect.left;
    mouse.y = e.clientY - rect.top;
    mouse.active = mouse.x >= 0 && mouse.x <= rect.width && mouse.y >= 0 && mouse.y <= rect.height;
  }

  function onPointerLeave() {
    mouse.active = false;
  }

  resize();
  renderFrame();
  window.addEventListener('resize', resize);
  window.addEventListener('pointermove', onPointerMove, { passive: true });
  canvas.addEventListener('pointerleave', onPointerLeave);

  return () => {
    if (rafId) cancelAnimationFrame(rafId);
    window.removeEventListener('resize', resize);
    window.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerleave', onPointerLeave);
  };
}
