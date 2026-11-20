/**
 * M5 60fps 초대형 와이드 글래스모피즘 시뮬레이션 Canvas 엔진 (v5 — 1600×540 Giant Stage)
 * - 1600×540 고해상도 와이드 비율로 노드(r=52), 가중치 배지(128×44), 글자(15~22px)를 큼직하고 선명하게 렌더링
 * - 각 노드 둘레에 현재 출력 비율(0~100%)을 보여주는 원형 아크 게이지와 글래스 하이라이트 적용
 * - 순전파(Blue →)와 역전파(← Orange) 다중 입자 스트림 및 5단계 합류 스파크 연출
 * - 17개 전 요소 마우스 호버 경로 추적, 클릭 시 상세 해설 인스펙터 연동, 2.2× 광학 돋보기(Loupe), 줌/팬
 */
import { fmt, sigmoidDerivativeFromOutput } from '../math/backprop.js';

const LOGICAL_W = 1600;
const LOGICAL_H = 540;
const NODE_R = 52;

const NODE_POS = {
  i1: { x: 150, y: 168, label: 'i1', sub: '입력 노드 1' },
  i2: { x: 150, y: 382, label: 'i2', sub: '입력 노드 2' },
  h1: { x: 555, y: 168, label: 'h1', sub: '은닉 노드 1' },
  h2: { x: 555, y: 382, label: 'h2', sub: '은닉 노드 2' },
  o1: { x: 980, y: 168, label: 'o1', sub: '출력 노드 1' },
  o2: { x: 980, y: 382, label: 'o2', sub: '출력 노드 2' },
};

const W1_DEFS = [
  { key: 'w1', from: 'i1', to: 'h1', r: 0, c: 0, t: 0.48 },
  { key: 'w2', from: 'i1', to: 'h2', r: 0, c: 1, t: 0.28 },
  { key: 'w3', from: 'i2', to: 'h1', r: 1, c: 0, t: 0.28 },
  { key: 'w4', from: 'i2', to: 'h2', r: 1, c: 1, t: 0.48 },
];

const W2_DEFS = [
  { key: 'w5', from: 'h1', to: 'o1', r: 0, c: 0, t: 0.50 },
  { key: 'w6', from: 'h1', to: 'o2', r: 0, c: 1, t: 0.30 },
  { key: 'w7', from: 'h2', to: 'o1', r: 1, c: 0, t: 0.30 },
  { key: 'w8', from: 'h2', to: 'o2', r: 1, c: 1, t: 0.50 },
];

const ERROR_BOXES = [
  { id: 'E_o1', oKey: 'o1', x: 1185, y: 92, w: 365, h: 146, cy: 168, idx: 0 },
  { id: 'E_o2', oKey: 'o2', x: 1185, y: 306, w: 365, h: 146, cy: 382, idx: 1 },
];
const TOTAL_BOX = { id: 'E_total', x: 1185, y: 468, w: 365, h: 50 };

function getThemePalette() {
  const isLight = document.documentElement.dataset.theme === 'light';
  if (isLight) {
    return {
      isLight: true,
      bgTop: '#f3f6fc',
      bgBot: '#e8eef7',
      orbFwd: 'rgba(37, 99, 235, 0.09)',
      orbBack: 'rgba(234, 88, 12, 0.08)',
      gridLine: 'rgba(15, 23, 42, 0.045)',
      heading: '#475569',
      textMain: '#0f172a',
      textSub: '#334155',
      textMuted: '#64748b',
      fwd: '#2563eb',
      fwdRgb: '37, 99, 235',
      fwdText: '#1d4ed8',
      fwdPillBg: 'rgba(239, 246, 255, 0.94)',
      back: '#ea580c',
      backRgb: '234, 88, 12',
      backAltRgb: '245, 158, 11',
      backText: '#c2410c',
      backPillBg: 'rgba(255, 247, 237, 0.95)',
      nodeFillTop: '#ffffff',
      nodeFillBot: '#edf2f9',
      nodeShadow: 'rgba(15, 23, 42, 0.12)',
      trackBg: 'rgba(15, 23, 42, 0.09)',
      pillBg: 'rgba(255, 255, 255, 0.92)',
      pillBorder: 'rgba(148, 163, 184, 0.45)',
      boxBg: 'rgba(255, 255, 255, 0.88)',
      boxFocusBg: 'rgba(255, 247, 237, 0.96)',
      boxBorder: 'rgba(148, 163, 184, 0.4)',
      loupeRing: '#1e293b',
      loupeShadow: 'rgba(15, 23, 42, 0.26)',
    };
  }
  return {
    isLight: false,
    bgTop: '#131c31',
    bgBot: '#0b1120',
    orbFwd: 'rgba(59, 130, 246, 0.14)',
    orbBack: 'rgba(249, 115, 22, 0.12)',
    gridLine: 'rgba(148, 163, 184, 0.06)',
    heading: '#94a3b8',
    textMain: '#f8fafc',
    textSub: '#cbd5e1',
    textMuted: '#94a3b8',
    fwd: '#3b82f6',
    fwdRgb: '59, 130, 246',
    fwdText: '#60a5fa',
    fwdPillBg: 'rgba(23, 37, 84, 0.92)',
    back: '#f97316',
    backRgb: '249, 115, 22',
    backAltRgb: '251, 191, 36',
    backText: '#fb923c',
    backPillBg: 'rgba(67, 20, 7, 0.92)',
    nodeFillTop: '#24324d',
    nodeFillBot: '#111a2e',
    nodeShadow: 'rgba(0, 0, 0, 0.5)',
    trackBg: 'rgba(255, 255, 255, 0.1)',
    pillBg: 'rgba(24, 34, 54, 0.9)',
    pillBorder: 'rgba(148, 163, 184, 0.3)',
    boxBg: 'rgba(22, 31, 50, 0.86)',
    boxFocusBg: 'rgba(54, 23, 14, 0.92)',
    boxBorder: 'rgba(148, 163, 184, 0.26)',
    loupeRing: '#f8fafc',
    loupeShadow: 'rgba(0, 0, 0, 0.65)',
  };
}

function pointToSegmentDist(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const lenSq = dx * dx + dy * dy || 1;
  const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / lenSq));
  return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
}

function drawRoundedRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

export function createNeuralCanvas(canvas, tooltipEl, { onSelectWeight, onInspectElement }) {
  const ctx = canvas.getContext('2d');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  let currentSim = null;
  let currentW1 = null;
  let currentW2 = null;
  let currentStage = 1;
  let selectedWeight = 'w5';
  let inspectedId = null;
  let flowPhase = 'auto';

  let loupeEnabled = false;
  let zoom = 1.0;
  let panX = 0;
  let panY = 0;
  let isDragging = false;
  let dragMoved = false;
  let dragStart = { x: 0, y: 0, panX: 0, panY: 0 };

  let hovered = null;
  let pulseEffects = [];
  let floatingTexts = []; // { x, y, text, color, alpha, vy }
  let shockwaves = []; // { x1, y1, x2, y2, color, prog }
  let pointer = { x: -999, y: -999, viewX: -999, viewY: -999, inside: false };
  const startTime = performance.now();
  let rafId = null;

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(LOGICAL_W * dpr);
    canvas.height = Math.round(LOGICAL_H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function triggerRipple(nodeKey, color) {
    const p = NODE_POS[nodeKey];
    if (!p) return;
    const pal = getThemePalette();
    pulseEffects.push({ x: p.x, y: p.y, color: color || pal.back, radius: NODE_R, alpha: 1.0 });
  }

  function spawnFloatingText(x, y, text, color = '#10b981') {
    floatingTexts.push({ x, y, text, color, alpha: 1.0, vy: -1.4 });
  }

  function triggerWeightUpdateEffect(weightKey, deltaWeight, errorReduction) {
    const def = W2_DEFS.find((w) => w.key === weightKey);
    const pal = getThemePalette();
    if (def) {
      const p1 = NODE_POS[def.from];
      const p2 = NODE_POS[def.to];
      const lx = p1.x + (p2.x - p1.x) * def.t;
      const ly = p1.y + (p2.y - p1.y) * def.t;
      const sign = deltaWeight >= 0 ? '+' : '';
      const dir = deltaWeight < 0 ? '↓' : '↑';
      spawnFloatingText(
        lx,
        ly - 28,
        `${weightKey} ${sign}${fmt(deltaWeight, 4)} ${dir}`,
        deltaWeight < 0 ? pal.fwd : pal.back
      );
      shockwaves.push({ x1: p2.x, y1: p2.y, x2: p1.x, y2: p1.y, color: pal.back, prog: 0 });
      triggerRipple(def.to, pal.back);
      triggerRipple(def.from, '#10b981');
    }
    if (errorReduction > 0) {
      spawnFloatingText(
        TOTAL_BOX.x + TOTAL_BOX.w / 2,
        TOTAL_BOX.y - 10,
        `오차 −${fmt(errorReduction, 5)} 감소!`,
        '#10b981'
      );
    }
  }

  function setState({ sim, W1, W2, stage, weightKey, phase = 'auto', activeInspectId }) {
    currentSim = sim;
    currentW1 = W1;
    currentW2 = W2;
    currentStage = stage;
    selectedWeight = weightKey;
    flowPhase = phase;
    if (activeInspectId !== undefined) inspectedId = activeInspectId;
  }

  function setLoupe(enabled) {
    loupeEnabled = Boolean(enabled);
    if (loupeEnabled && tooltipEl) tooltipEl.hidden = true;
  }

  function setZoom(nextZoom) {
    zoom = Math.max(1.0, Math.min(1.8, nextZoom));
    if (zoom <= 1.01) {
      zoom = 1.0;
      panX = 0;
      panY = 0;
    } else {
      clampPan();
    }
    return zoom;
  }

  function clampPan() {
    const maxX = ((zoom - 1) * LOGICAL_W) / 2;
    const maxY = ((zoom - 1) * LOGICAL_H) / 2;
    panX = Math.max(-maxX, Math.min(maxX, panX));
    panY = Math.max(-maxY, Math.min(maxY, panY));
  }

  function screenToWorld(e) {
    const rect = canvas.getBoundingClientRect();
    const viewX = ((e.clientX - rect.left) / rect.width) * LOGICAL_W;
    const viewY = ((e.clientY - rect.top) / rect.height) * LOGICAL_H;
    const cx = LOGICAL_W / 2;
    const cy = LOGICAL_H / 2;
    const worldX = (viewX - cx - panX) / zoom + cx;
    const worldY = (viewY - cy - panY) / zoom + cy;
    return {
      x: worldX,
      y: worldY,
      viewX,
      viewY,
      clientX: e.clientX - rect.left,
      clientY: e.clientY - rect.top,
    };
  }

  function hitTest(wx, wy) {
    for (const [key, pos] of Object.entries(NODE_POS)) {
      if (Math.hypot(wx - pos.x, wy - pos.y) <= NODE_R + 4) {
        return { kind: 'node', id: key };
      }
    }
    for (const box of ERROR_BOXES) {
      if (wx >= box.x && wx <= box.x + box.w && wy >= box.y && wy <= box.y + box.h) {
        return { kind: 'error', id: box.id };
      }
    }
    if (
      wx >= TOTAL_BOX.x &&
      wx <= TOTAL_BOX.x + TOTAL_BOX.w &&
      wy >= TOTAL_BOX.y &&
      wy <= TOTAL_BOX.y + TOTAL_BOX.h
    ) {
      return { kind: 'error', id: TOTAL_BOX.id };
    }
    for (const def of W2_DEFS) {
      const p1 = NODE_POS[def.from];
      const p2 = NODE_POS[def.to];
      const lx = p1.x + (p2.x - p1.x) * def.t;
      const ly = p1.y + (p2.y - p1.y) * def.t;
      if (Math.abs(wx - lx) <= 70 && Math.abs(wy - ly) <= 26) {
        return { kind: 'edge', id: def.key, layer: 2 };
      }
      if (pointToSegmentDist(wx, wy, p1.x + NODE_R, p1.y, p2.x - NODE_R, p2.y) <= 18) {
        return { kind: 'edge', id: def.key, layer: 2 };
      }
    }
    for (const def of W1_DEFS) {
      const p1 = NODE_POS[def.from];
      const p2 = NODE_POS[def.to];
      const lx = p1.x + (p2.x - p1.x) * def.t;
      const ly = p1.y + (p2.y - p1.y) * def.t;
      if (Math.abs(wx - lx) <= 62 && Math.abs(wy - ly) <= 24) {
        return { kind: 'edge', id: def.key, layer: 1 };
      }
      if (pointToSegmentDist(wx, wy, p1.x + NODE_R, p1.y, p2.x - NODE_R, p2.y) <= 16) {
        return { kind: 'edge', id: def.key, layer: 1 };
      }
    }
    return null;
  }

  function isConnectedToHover(entityType, defOrKey) {
    if (!hovered) return false;
    if (hovered.kind === 'node') {
      if (entityType === 'node') return hovered.id === defOrKey;
      if (entityType === 'edge') return defOrKey.from === hovered.id || defOrKey.to === hovered.id;
      if (entityType === 'error') return defOrKey === `E_${hovered.id}`;
    }
    if (hovered.kind === 'edge') {
      if (entityType === 'edge') return hovered.id === defOrKey.key;
      const allDefs = [...W1_DEFS, ...W2_DEFS];
      const found = allDefs.find((d) => d.key === hovered.id);
      if (found && entityType === 'node') {
        return defOrKey === found.from || defOrKey === found.to;
      }
    }
    if (hovered.kind === 'error') {
      if (entityType === 'error') return hovered.id === defOrKey || hovered.id === 'E_total';
      if (entityType === 'node') {
        if (hovered.id === 'E_o1') return defOrKey === 'o1';
        if (hovered.id === 'E_o2') return defOrKey === 'o2';
        return defOrKey === 'o1' || defOrKey === 'o2';
      }
    }
    return false;
  }

  function updateTooltip(pt) {
    if (!tooltipEl || !currentSim || loupeEnabled || isDragging) {
      if (tooltipEl) tooltipEl.hidden = true;
      return;
    }
    if (!hovered) {
      tooltipEl.hidden = true;
      return;
    }

    const b = currentSim.before;
    let html = '';

    if (hovered.kind === 'edge') {
      if (currentSim.allDetails[hovered.id]) {
        const d = currentSim.allDetails[hovered.id];
        const dir = d.deltaWeight < 0 ? '감소 방향 ↓' : '증가 방향 ↑';
        html = `
          <div class="tt-title">${d.meta.label} <span class="tt-hint">클릭해 계산 과정 열기</span></div>
          <div>현재 가중치 <strong class="num">${fmt(d.oldWeight, 4)}</strong> → 수정 후 <strong class="num">${fmt(d.newWeight, 4)}</strong></div>
          <div>신호 전달 <strong class="num">${fmt(d.hiddenVal * d.oldWeight, 4)}</strong> · 기울기 <strong class="num">${fmt(d.gradient, 5)}</strong> (${dir})</div>
        `;
      } else {
        const def = W1_DEFS.find((w) => w.key === hovered.id);
        const wVal = currentW1[def.r][def.c];
        const inVal = b.inputs[def.r];
        html = `
          <div class="tt-title">${def.key} (${def.from} → ${def.to}) <span class="tt-hint">클릭해 계산 과정 열기</span></div>
          <div>가중치 <strong class="num">${fmt(wVal, 2)}</strong> · 전달 신호 <strong class="num">${fmt(inVal * wVal, 5)}</strong></div>
        `;
      }
    } else if (hovered.kind === 'node') {
      const k = hovered.id;
      const lines = {
        i1: `입력값 <strong class="num">${fmt(b.inputs[0], 2)}</strong> · w1, w2로 신호를 보냄`,
        i2: `입력값 <strong class="num">${fmt(b.inputs[1], 2)}</strong> · w3, w4로 신호를 보냄`,
        h1: `은닉 출력 <strong class="num">${fmt(b.h[0], 6)}</strong> (가중합 net=${fmt(b.netH[0], 4)})`,
        h2: `은닉 출력 <strong class="num">${fmt(b.h[1], 6)}</strong> (가중합 net=${fmt(b.netH[1], 4)})`,
        o1: `출력 <strong class="num">${fmt(b.o[0], 6)}</strong> · 목표 ${fmt(b.targets[0], 2)} · 민감도 ${fmt(sigmoidDerivativeFromOutput(b.o[0]), 4)}`,
        o2: `출력 <strong class="num">${fmt(b.o[1], 6)}</strong> · 목표 ${fmt(b.targets[1], 2)} · 민감도 ${fmt(sigmoidDerivativeFromOutput(b.o[1]), 4)}`,
      };
      html = `
        <div class="tt-title">노드 ${k.toUpperCase()} <span class="tt-hint">클릭해 상세 해부</span></div>
        <div>${lines[k]}</div>
      `;
    } else if (hovered.kind === 'error') {
      const val =
        hovered.id === 'E_o1'
          ? fmt(b.errors[0], 6)
          : hovered.id === 'E_o2'
            ? fmt(b.errors[1], 6)
            : fmt(b.eTotal, 6);
      html = `
        <div class="tt-title">${hovered.id} = <span class="num">${val}</span> <span class="tt-hint">클릭해 공식 해부</span></div>
        <div>오차 제곱 공식과 0.5를 곱하는 이유를 상세 패널에서 확인합니다.</div>
      `;
    }

    tooltipEl.hidden = false;
    tooltipEl.innerHTML = html;
    tooltipEl.style.left = `${Math.min(Math.max(12, pt.clientX + 18), canvas.clientWidth - 310)}px`;
    tooltipEl.style.top = `${Math.max(12, pt.clientY - 82)}px`;
  }

  function drawStreamParticles(x1, y1, x2, y2, count, speed, colorRgb, radius, reverse, timeSec, alphaScale, maxRange, pal) {
    for (let i = 0; i < count; i++) {
      const prog = (timeSec * speed + i / count) % 1;
      const t = reverse ? 1 - prog * maxRange : prog * maxRange;
      const px = x1 + (x2 - x1) * t;
      const py = y1 + (y2 - y1) * t;

      const tailShift = 0.095 * maxRange;
      const tailT = reverse ? Math.min(1, t + tailShift) : Math.max(0, t - tailShift);
      const tx = x1 + (x2 - x1) * tailT;
      const ty = y1 + (y2 - y1) * tailT;

      const grad = ctx.createLinearGradient(tx, ty, px, py);
      grad.addColorStop(0, `rgba(${colorRgb}, 0)`);
      grad.addColorStop(1, `rgba(${colorRgb}, ${(0.96 * alphaScale).toFixed(2)})`);

      ctx.strokeStyle = grad;
      ctx.lineWidth = radius * 1.9;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(px, py);
      ctx.stroke();

      // 외부 광채 + 코어
      ctx.fillStyle = `rgba(${colorRgb}, ${(0.4 * alphaScale).toFixed(2)})`;
      ctx.beginPath();
      ctx.arc(px, py, radius * 1.6, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = pal.isLight ? `rgba(${colorRgb}, 1)` : '#ffffff';
      ctx.beginPath();
      ctx.arc(px, py, radius * 0.85, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function renderScene(timeSec, pal, xray = false) {
    // 1) 글래스모피즘 오로라 배경 & 정밀 그리드
    const bgGrad = ctx.createLinearGradient(0, 0, LOGICAL_W, LOGICAL_H);
    bgGrad.addColorStop(0, pal.bgTop);
    bgGrad.addColorStop(1, pal.bgBot);
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, LOGICAL_W, LOGICAL_H);

    // 은은한 앰비언트 오로라 글로우
    const orb1 = ctx.createRadialGradient(360, 180, 20, 360, 180, 420);
    orb1.addColorStop(0, pal.orbFwd);
    orb1.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = orb1;
    ctx.fillRect(0, 0, LOGICAL_W, LOGICAL_H);

    const orb2 = ctx.createRadialGradient(1120, 340, 20, 1120, 340, 460);
    orb2.addColorStop(0, pal.orbBack);
    orb2.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = orb2;
    ctx.fillRect(0, 0, LOGICAL_W, LOGICAL_H);

    // 미세 그리드 라인
    ctx.strokeStyle = pal.gridLine;
    ctx.lineWidth = 1;
    const gridStep = 48;
    ctx.beginPath();
    for (let gx = gridStep; gx < LOGICAL_W; gx += gridStep) {
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, LOGICAL_H);
    }
    for (let gy = gridStep; gy < LOGICAL_H; gy += gridStep) {
      ctx.moveTo(0, gy);
      ctx.lineTo(LOGICAL_W, gy);
    }
    ctx.stroke();

    if (!currentSim) return;

    const { before, selected, allDetails } = currentSim;
    const meta = selected.meta;
    const st = currentStage;
    const isBackwardMode = flowPhase === 'backward' || (flowPhase === 'auto' && st >= 3);
    const isForwardMode = flowPhase === 'forward' || (flowPhase === 'auto' && st <= 2);

    // 상단 층 헤더 배지
    ctx.font = '800 16px "Pretendard", "Malgun Gothic", sans-serif';
    ctx.fillStyle = pal.heading;
    ctx.textAlign = 'center';
    ctx.fillText('입력층 (Input)', 150, 42);
    ctx.fillText('은닉층 (Hidden · Sigmoid)', 555, 42);
    ctx.fillText('출력층 (Output · Sigmoid)', 980, 42);
    ctx.fillText('목표 · 오차 비교 (Loss)', 1367, 42);

    // 2) W1 연결선 + 글래스 가중치 배지
    for (const def of W1_DEFS) {
      const p1 = NODE_POS[def.from];
      const p2 = NODE_POS[def.to];
      const wVal = currentW1[def.r][def.c];
      const x1 = p1.x + NODE_R;
      const y1 = p1.y;
      const x2 = p2.x - NODE_R;
      const y2 = p2.y;
      const hovRel = isConnectedToHover('edge', def);
      const isInsp = inspectedId === def.key;
      const connectedToSelectedH = def.to === meta.from;

      ctx.save();
      const lineW = 3.0 + Math.min(4.5, Math.abs(wVal) * 6.5) + (hovRel || isInsp ? 2.5 : 0);

      if (isBackwardMode) {
        const backAlpha = hovRel || isInsp ? 0.95 : connectedToSelectedH && st >= 5 ? 0.62 : 0.24;
        ctx.strokeStyle = `rgba(${pal.backRgb}, ${backAlpha})`;
        ctx.lineWidth = lineW;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();

        if (connectedToSelectedH && st >= 5) {
          drawStreamParticles(x1, y1, x2, y2, 3, 0.34, pal.backRgb, 3.6, true, timeSec, 0.72, 1, pal);
        }
      } else {
        const fwdAlpha = hovRel || isInsp ? 1.0 : st === 1 ? 0.85 : 0.52;
        ctx.strokeStyle = `rgba(${pal.fwdRgb}, ${fwdAlpha})`;
        ctx.lineWidth = lineW;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();

        drawStreamParticles(x1, y1, x2, y2, st === 1 || hovRel ? 5 : 3, 0.3 + Math.abs(wVal) * 0.35, pal.fwdRgb, st === 1 || hovRel ? 4.4 : 3.2, false, timeSec, st === 1 || hovRel ? 1.0 : 0.7, 1, pal);
      }

      // 큼직한 W1 글래스 배지 (110 × 40)
      const lx = p1.x + (p2.x - p1.x) * def.t;
      const ly = p1.y + (p2.y - p1.y) * def.t;
      ctx.globalAlpha = !hovRel && !isInsp && isBackwardMode && !(connectedToSelectedH && st >= 5) ? 0.55 : 1;
      drawRoundedRect(ctx, lx - 56, ly - 20, 112, 40, 11);
      ctx.fillStyle = hovRel || isInsp ? pal.fwdPillBg : pal.pillBg;
      ctx.fill();
      ctx.strokeStyle = hovRel || isInsp ? pal.fwd : pal.pillBorder;
      ctx.lineWidth = hovRel || isInsp ? 2.4 : 1.4;
      ctx.stroke();

      ctx.font = '800 15.5px ui-monospace, SFMono-Regular, monospace';
      ctx.fillStyle = hovRel || isInsp ? pal.fwdText : pal.textMain;
      ctx.textAlign = 'center';
      ctx.fillText(`${def.key} = ${fmt(wVal, 2)}`, lx, ly - 1);

      const trans = before.inputs[def.r] * wVal;
      ctx.font = '600 11.5px "Pretendard", sans-serif';
      ctx.fillStyle = pal.textMuted;
      ctx.fillText(`전달 ${fmt(trans, 4)}`, lx, ly + 14);
      ctx.restore();
    }

    // 3) W2 연결선 (순전파 Blue → / 역전파 Orange ←) + 대형 2줄 글래스 배지
    for (const def of W2_DEFS) {
      const p1 = NODE_POS[def.from];
      const p2 = NODE_POS[def.to];
      const wVal = currentW2[def.r][def.c];
      const d = allDetails[def.key];
      const isSel = def.key === selectedWeight;
      const hovRel = isConnectedToHover('edge', def);
      const isInsp = inspectedId === def.key;
      const dimmed = isBackwardMode && !isSel && !hovRel && !isInsp;

      const x1 = p1.x + NODE_R;
      const y1 = p1.y;
      const x2 = p2.x - NODE_R;
      const y2 = p2.y;
      const baseWidth = 3.5 + Math.min(6.0, Math.abs(wVal) * 6.5);

      ctx.save();
      if (isBackwardMode && (isSel || hovRel || isInsp)) {
        ctx.shadowColor = pal.back;
        ctx.shadowBlur = pal.isLight ? 12 : 20;
        ctx.strokeStyle = pal.back;
        ctx.lineWidth = baseWidth + (isSel ? 3.2 : 2.0);
      } else if (isForwardMode && (isSel || hovRel || isInsp)) {
        ctx.shadowColor = pal.fwd;
        ctx.shadowBlur = pal.isLight ? 12 : 20;
        ctx.strokeStyle = pal.fwd;
        ctx.lineWidth = baseWidth + 2.4;
      } else if (isBackwardMode) {
        ctx.strokeStyle = `rgba(${pal.backRgb}, 0.28)`;
        ctx.lineWidth = baseWidth * 0.9;
      } else {
        ctx.strokeStyle = `rgba(${pal.fwdRgb}, 0.65)`;
        ctx.lineWidth = baseWidth;
      }

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      ctx.shadowBlur = 0;

      if (isForwardMode) {
        const speed = 0.3 + Math.abs(wVal) * 0.38;
        drawStreamParticles(x1, y1, x2, y2, isSel || st === 1 || hovRel ? 6 : 3, speed, pal.fwdRgb, isSel || hovRel ? 4.8 : 3.4, false, timeSec, isSel || st === 1 || hovRel ? 1.0 : 0.68, 1, pal);
      } else if (isSel) {
        const backSpeed = st === 3 ? 0.35 : st === 4 ? 0.45 : 0.56 + Math.min(0.42, Math.abs(selected.gradient) * 3.0);
        if (st === 5) {
          drawStreamParticles(x1, y1, x2, y2, 5, backSpeed, pal.backRgb, 5.2, true, timeSec, 1.0, 0.5, pal);
          drawStreamParticles(x1, y1, x2, y2, 5, 0.48, pal.fwdRgb, 4.8, false, timeSec, 1.0, 0.5, pal);
          const mx = (x1 + x2) * 0.5;
          const my = (y1 + y2) * 0.5;
          const sparkR = 24 + Math.sin(timeSec * 8) * 6;
          const sg = ctx.createRadialGradient(mx, my, 3, mx, my, sparkR);
          sg.addColorStop(0, 'rgba(255, 255, 255, 0.95)');
          sg.addColorStop(0.5, `rgba(${pal.backRgb}, 0.6)`);
          sg.addColorStop(1, `rgba(${pal.fwdRgb}, 0)`);
          ctx.fillStyle = sg;
          ctx.beginPath();
          ctx.arc(mx, my, sparkR, 0, Math.PI * 2);
          ctx.fill();
        } else {
          drawStreamParticles(x1, y1, x2, y2, 6, backSpeed, pal.backRgb, 5.2, true, timeSec, st >= 5 ? 1.0 : 0.88, 1, pal);
        }
      } else {
        drawStreamParticles(x1, y1, x2, y2, 3, 0.26, pal.backAltRgb, 3.0, true, timeSec, hovRel ? 0.85 : 0.35, 1, pal);
      }

      // 큼직한 W2 2줄 글래스모피즘 배지 (138 × 46)
      const lx = p1.x + (p2.x - p1.x) * def.t;
      const ly = p1.y + (p2.y - p1.y) * def.t;
      const bw = isSel ? 148 : 132;
      const bh = isSel ? 48 : 44;
      ctx.globalAlpha = dimmed ? 0.55 : 1;

      ctx.shadowColor = pal.nodeShadow;
      ctx.shadowBlur = isSel || hovRel ? 16 : 8;
      drawRoundedRect(ctx, lx - bw / 2, ly - bh / 2, bw, bh, 12);
      ctx.fillStyle = isBackwardMode && (isSel || hovRel || isInsp)
        ? pal.backPillBg
        : isForwardMode && (isSel || hovRel || isInsp)
          ? pal.fwdPillBg
          : pal.pillBg;
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.strokeStyle = isBackwardMode && (isSel || hovRel || isInsp)
        ? pal.back
        : isForwardMode && (isSel || hovRel || isInsp)
          ? pal.fwd
          : pal.pillBorder;
      ctx.lineWidth = isSel || hovRel || isInsp ? 2.6 : 1.4;
      ctx.stroke();

      ctx.font = '800 16.5px ui-monospace, SFMono-Regular, monospace';
      ctx.fillStyle = isBackwardMode && (isSel || hovRel || isInsp)
        ? pal.backText
        : isForwardMode && (isSel || hovRel || isInsp)
          ? pal.fwdText
          : pal.textMain;
      ctx.textAlign = 'center';
      ctx.fillText(`${def.key} = ${fmt(wVal, isSel ? 4 : 2)}`, lx, ly - 3);

      ctx.font = '600 12px "Pretendard", sans-serif';
      ctx.fillStyle = isBackwardMode ? pal.backText : pal.textMuted;
      const subText = isBackwardMode
        ? `기울기 ${d.gradient >= 0 ? '+' : ''}${fmt(d.gradient, 4)}`
        : `${def.from}→${def.to} (${fmt(d.hiddenVal * d.oldWeight, 3)})`;
      ctx.fillText(subText, lx, ly + 15);
      ctx.restore();
    }

    // 4) 출력층 <-> 목표 & 오차 대형 글래스 카드 (우측)
    for (const box of ERROR_BOXES) {
      const target = before.targets[box.idx];
      const out = before.o[box.idx];
      const err = before.errors[box.idx];
      const hovRel = isConnectedToHover('error', box.id);
      const isInsp = inspectedId === box.id;
      const isFocus = st === 2 || (isBackwardMode && meta.to === box.oKey) || hovRel || isInsp;
      const dimmed = isBackwardMode && meta.to !== box.oKey && !hovRel && !isInsp;

      ctx.save();
      ctx.strokeStyle = isFocus ? pal.back : dimmed ? pal.pillBorder : `rgba(${pal.backRgb}, 0.45)`;
      ctx.lineWidth = isFocus ? 4.2 : 2.2;
      if (!isFocus) ctx.setLineDash([7, 6]);
      ctx.beginPath();
      ctx.moveTo(980 + NODE_R, box.cy);
      ctx.lineTo(box.x - 4, box.cy);
      ctx.stroke();
      ctx.setLineDash([]);

      if (isBackwardMode && (meta.to === box.oKey || hovRel)) {
        drawStreamParticles(980 + NODE_R + 4, box.cy, box.x - 6, box.cy, 5, 0.72, pal.backRgb, 4.8, true, timeSec, 1.0, 1, pal);
      } else if (st === 2) {
        drawStreamParticles(980 + NODE_R + 4, box.cy, box.x - 6, box.cy, 4, 0.45, pal.backAltRgb, 3.8, false, timeSec, 0.88, 1, pal);
      }

      ctx.globalAlpha = dimmed ? 0.52 : 1;
      ctx.shadowColor = pal.nodeShadow;
      ctx.shadowBlur = isFocus ? 20 : 10;
      drawRoundedRect(ctx, box.x, box.y, box.w, box.h, 16);
      ctx.fillStyle = isFocus ? pal.boxFocusBg : pal.boxBg;
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.strokeStyle = isFocus ? pal.back : pal.boxBorder;
      ctx.lineWidth = isFocus ? 2.6 : 1.5;
      ctx.stroke();

      // 카드 내부 텍스트 (큼직한 16~18px)
      ctx.textAlign = 'left';
      ctx.font = '700 16px "Pretendard", sans-serif';
      ctx.fillStyle = pal.textSub;
      ctx.fillText(`목표 ${box.oKey}`, box.x + 22, box.y + 34);
      ctx.fillText(`출력 ${box.oKey}`, box.x + 22, box.y + 66);

      ctx.textAlign = 'right';
      ctx.font = '800 18px ui-monospace, SFMono-Regular, monospace';
      ctx.fillStyle = pal.textMain;
      ctx.fillText(fmt(target, 2), box.x + box.w - 22, box.y + 34);
      ctx.fillStyle = isBackwardMode ? pal.backText : pal.fwdText;
      ctx.fillText(fmt(out, 4), box.x + box.w - 22, box.y + 66);

      // 카드 내 시각적 오차 간극 바
      const barX = box.x + 22;
      const barY = box.y + 86;
      const barW = box.w - 44;
      drawRoundedRect(ctx, barX, barY, barW, 10, 5);
      ctx.fillStyle = pal.trackBg;
      ctx.fill();

      const minP = Math.min(target, out);
      const spanP = Math.max(0.015, Math.abs(target - out));
      drawRoundedRect(ctx, barX + minP * barW, barY, spanP * barW, 10, 5);
      ctx.fillStyle = pal.back;
      ctx.fill();

      ctx.textAlign = 'left';
      ctx.font = '800 16.5px ui-monospace, SFMono-Regular, monospace';
      ctx.fillStyle = pal.backText;
      ctx.fillText(`개별 오차 E_${box.oKey} = ${fmt(err, 4)}`, box.x + 22, box.y + 126);
      ctx.restore();
    }

    // 전체 오차 대형 배지
    {
      const hovTotal = isConnectedToHover('error', 'E_total') || inspectedId === 'E_total';
      const emph = st === 2 || st === 6 || hovTotal;
      ctx.save();
      ctx.shadowColor = pal.nodeShadow;
      ctx.shadowBlur = 12;
      drawRoundedRect(ctx, TOTAL_BOX.x, TOTAL_BOX.y, TOTAL_BOX.w, TOTAL_BOX.h, 13);
      ctx.fillStyle = emph ? pal.boxFocusBg : pal.boxBg;
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = emph ? pal.back : pal.boxBorder;
      ctx.lineWidth = emph ? 2.4 : 1.5;
      ctx.stroke();
      ctx.textAlign = 'center';
      ctx.font = '800 18.5px ui-monospace, SFMono-Regular, monospace';
      ctx.fillStyle = emph ? pal.backText : pal.textMain;
      ctx.fillText(`전체 오차 E_total = ${fmt(before.eTotal, 6)}`, TOTAL_BOX.x + TOTAL_BOX.w / 2, TOTAL_BOX.y + 32);
      ctx.restore();
    }

    // 5) 충격파(Ripple) · 레이저 빔(Shockwave) · 플로팅 수치 애니메이션
    if (!xray) {
      pulseEffects = pulseEffects.filter((p) => p.alpha > 0.02);
      for (const p of pulseEffects) {
        ctx.save();
        ctx.strokeStyle = p.color;
        ctx.globalAlpha = p.alpha;
        ctx.lineWidth = 3.5;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
        p.radius += 3.1;
        p.alpha *= 0.9;
      }

      shockwaves = shockwaves.filter((s) => s.prog < 1);
      for (const s of shockwaves) {
        const sx = s.x1 + (s.x2 - s.x1) * s.prog;
        const sy = s.y1 + (s.y2 - s.y1) * s.prog;
        ctx.save();
        ctx.fillStyle = s.color;
        ctx.shadowColor = s.color;
        ctx.shadowBlur = 22;
        ctx.beginPath();
        ctx.arc(sx, sy, 10 * (1 - s.prog * 0.4), 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        s.prog += 0.065;
      }

      floatingTexts = floatingTexts.filter((ft) => ft.alpha > 0.03);
      for (const ft of floatingTexts) {
        ctx.save();
        ctx.globalAlpha = ft.alpha;
        ctx.font = '900 18px ui-monospace, SFMono-Regular, monospace';
        ctx.fillStyle = ft.color;
        ctx.textAlign = 'center';
        ctx.shadowColor = pal.nodeShadow;
        ctx.shadowBlur = 10;
        ctx.fillText(ft.text, ft.x, ft.y);
        ctx.restore();
        ft.y += ft.vy;
        ft.alpha *= 0.96;
      }
    }

    // 6) 대형 뉴모피즘/글래스 뉴런 노드 렌더링 (r = 52)
    const rawValMap = {
      i1: before.inputs[0],
      i2: before.inputs[1],
      h1: before.h[0],
      h2: before.h[1],
      o1: before.o[0],
      o2: before.o[1],
    };
    const valMap = {
      i1: fmt(before.inputs[0], 2),
      i2: fmt(before.inputs[1], 2),
      h1: fmt(before.h[0], 4),
      h2: fmt(before.h[1], 4),
      o1: fmt(before.o[0], 4),
      o2: fmt(before.o[1], 4),
    };
    const subDetailMap = {
      i1: '입력 신호 1',
      i2: '입력 신호 2',
      h1: `은닉 1 · net ${fmt(before.netH[0], 3)}`,
      h2: `은닉 2 · net ${fmt(before.netH[1], 3)}`,
      o1: `출력 1 · 민감도 ${fmt(sigmoidDerivativeFromOutput(before.o[0]), 3)}`,
      o2: `출력 2 · 민감도 ${fmt(sigmoidDerivativeFromOutput(before.o[1]), 3)}`,
    };

    for (const [key, pos] of Object.entries(NODE_POS)) {
      const hovRel = isConnectedToHover('node', key);
      const isInsp = inspectedId === key;
      let focusColor = isBackwardMode ? pal.back : pal.fwd;
      let focusText = isBackwardMode ? pal.backText : pal.fwdText;
      let dimmed = false;
      let isPulse = false;

      if (key.startsWith('i')) {
        focusColor = pal.fwd;
        focusText = pal.fwdText;
        if (isBackwardMode) dimmed = true;
      } else if (key.startsWith('h')) {
        if (isBackwardMode) {
          if (meta.from === key) {
            focusColor = st === 5 ? pal.fwd : pal.back;
            focusText = st === 5 ? pal.fwdText : pal.backText;
            isPulse = st >= 5;
          } else {
            dimmed = true;
          }
        } else {
          focusColor = pal.fwd;
          focusText = pal.fwdText;
        }
      } else if (key.startsWith('o')) {
        if (isBackwardMode) {
          if (meta.to === key) {
            focusColor = pal.back;
            focusText = pal.backText;
            isPulse = st === 3 || st === 4;
          } else {
            dimmed = true;
          }
        } else {
          focusColor = pal.fwd;
          focusText = pal.fwdText;
        }
      }

      ctx.save();
      ctx.globalAlpha = dimmed && !hovRel && !isInsp ? 0.5 : 1;

      // 맥동 외곽 링
      if (isPulse || isInsp || hovRel) {
        const breath = NODE_R + 11 + Math.sin(timeSec * 5) * 4;
        ctx.strokeStyle = focusColor;
        ctx.lineWidth = isInsp ? 3.2 : 2.4;
        ctx.setLineDash([7, 5]);
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, breath, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // 뉴모피즘 깊이 그림자
      ctx.shadowColor = hovRel || isPulse || isInsp ? focusColor : pal.nodeShadow;
      ctx.shadowBlur = hovRel || isPulse || isInsp ? 26 : 16;
      ctx.shadowOffsetY = 6;

      const r = hovRel || isInsp ? NODE_R + 4 : NODE_R;
      const nodeGrad = ctx.createRadialGradient(pos.x - 16, pos.y - 18, 6, pos.x, pos.y, r + 4);
      nodeGrad.addColorStop(0, pal.nodeFillTop);
      nodeGrad.addColorStop(1, pal.nodeFillBot);
      ctx.fillStyle = nodeGrad;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.shadowOffsetY = 0;

      // 노드 테두리 트랙
      ctx.strokeStyle = pal.pillBorder;
      ctx.lineWidth = 3.5;
      ctx.stroke();

      // 노드 활성도 원형 아크 게이지 (0~100%)
      const ratio = Math.max(0.04, Math.min(1, rawValMap[key]));
      ctx.strokeStyle = focusColor;
      ctx.lineWidth = 5.0;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * ratio);
      ctx.stroke();

      // 노드 내부 텍스트 (22px 제목 + 19px 값)
      ctx.textAlign = 'center';
      ctx.font = '900 22px "Pretendard", sans-serif';
      ctx.fillStyle = pal.textMain;
      ctx.fillText(pos.label, pos.x, pos.y - 7);

      ctx.font = '800 18.5px ui-monospace, SFMono-Regular, monospace';
      ctx.fillStyle = focusText;
      ctx.fillText(valMap[key], pos.x, pos.y + 19);

      // 노드 하단 캡션 필
      ctx.font = '700 13.5px "Pretendard", sans-serif';
      ctx.fillStyle = xray ? focusText : pal.textSub;
      ctx.fillText(subDetailMap[key], pos.x, pos.y + r + 26);
      ctx.restore();
    }

    // 7) 하단 상태 안내 & 범례
    if (!xray) {
      ctx.save();
      const statusText = isForwardMode
        ? st === 2
          ? '오차 계산: 출력과 목표의 차이를 측정합니다 — 노드·가중치·오차 카드를 클릭하면 상세 계산식이 열립니다'
          : '순전파 (파랑 →): 입력 신호가 가중치를 거쳐 출력으로 흐릅니다 — 아무 요소나 클릭해 내부 계산을 해부해 보세요'
        : st === 5
          ? `역전파 3단계: ${meta.from}의 파란 신호와 ${meta.to}의 주황 오차가 ${meta.key}에서 만나 곱해집니다`
          : `역전파 (← 주황): 오차가 ${meta.to} → ${meta.key} → ${meta.from} 방향으로 거슬러 올라가며 가중치를 수정합니다`;

      ctx.font = '700 15px "Pretendard", sans-serif';
      ctx.fillStyle = isBackwardMode ? pal.backText : pal.fwdText;
      ctx.textAlign = 'left';
      ctx.fillText(statusText, 36, LOGICAL_H - 20);

      ctx.font = '700 14px "Pretendard", sans-serif';
      ctx.fillStyle = pal.fwd;
      ctx.beginPath();
      ctx.arc(935, LOGICAL_H - 25, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = pal.textSub;
      ctx.fillText('순전파 (신호 전달 →)', 948, LOGICAL_H - 20);

      ctx.fillStyle = pal.back;
      ctx.beginPath();
      ctx.arc(1098, LOGICAL_H - 25, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = pal.textSub;
      ctx.fillText('← 역전파 (오차 역류)', 1111, LOGICAL_H - 20);
      ctx.restore();
    }
  }

  function drawLoupe(timeSec, pal) {
    if (!loupeEnabled || !pointer.inside) return;

    const r = 132;
    const mag = 2.15;
    const lx = pointer.viewX;
    const ly = pointer.viewY;

    ctx.save();
    ctx.shadowColor = pal.loupeShadow;
    ctx.shadowBlur = 30;
    ctx.beginPath();
    ctx.arc(lx, ly, r, 0, Math.PI * 2);
    ctx.fillStyle = pal.bgBot;
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.beginPath();
    ctx.arc(lx, ly, r, 0, Math.PI * 2);
    ctx.clip();

    ctx.translate(lx, ly);
    ctx.scale(mag, mag);
    ctx.translate(-pointer.x, -pointer.y);
    renderScene(timeSec, pal, true);
    ctx.restore();

    ctx.save();
    ctx.strokeStyle = pal.loupeRing;
    ctx.lineWidth = 3.5;
    ctx.beginPath();
    ctx.arc(lx, ly, r, 0, Math.PI * 2);
    ctx.stroke();

    ctx.strokeStyle = pal.isLight ? 'rgba(37, 99, 235, 0.35)' : 'rgba(96, 165, 250, 0.45)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(lx - 14, ly);
    ctx.lineTo(lx + 14, ly);
    ctx.moveTo(lx, ly - 14);
    ctx.lineTo(lx, ly + 14);
    ctx.stroke();

    drawRoundedRect(ctx, lx - 52, ly + r - 15, 104, 26, 13);
    ctx.fillStyle = pal.fwd;
    ctx.fill();
    ctx.font = '800 12.5px ui-monospace, SFMono-Regular, monospace';
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.fillText('2.2× 돋보기', lx, ly + r + 2);
    ctx.restore();
  }

  function draw(now) {
    const timeSec = reducedMotion ? 0.35 : (now - startTime) / 1000;
    const pal = getThemePalette();

    ctx.save();
    ctx.clearRect(0, 0, LOGICAL_W, LOGICAL_H);

    if (zoom > 1.001) {
      ctx.translate(LOGICAL_W / 2 + panX, LOGICAL_H / 2 + panY);
      ctx.scale(zoom, zoom);
      ctx.translate(-LOGICAL_W / 2, -LOGICAL_H / 2);
    }
    renderScene(timeSec, pal, zoom >= 1.45);
    ctx.restore();

    drawLoupe(timeSec, pal);

    rafId = requestAnimationFrame(draw);
  }

  resize();
  rafId = requestAnimationFrame(draw);

  canvas.addEventListener('pointerdown', (e) => {
    const pt = screenToWorld(e);
    if (zoom > 1.01) {
      isDragging = true;
      dragMoved = false;
      dragStart = { x: pt.viewX, y: pt.viewY, panX, panY };
      canvas.style.cursor = 'grabbing';
    }
  });

  window.addEventListener('pointerup', () => {
    if (isDragging) {
      isDragging = false;
      canvas.style.cursor = hovered ? 'pointer' : zoom > 1.01 ? 'grab' : 'default';
    }
  });

  canvas.addEventListener('pointermove', (e) => {
    const pt = screenToWorld(e);
    pointer = { x: pt.x, y: pt.y, viewX: pt.viewX, viewY: pt.viewY, inside: true };

    if (isDragging) {
      const dx = pt.viewX - dragStart.x;
      const dy = pt.viewY - dragStart.y;
      if (Math.hypot(dx, dy) > 4) dragMoved = true;
      panX = dragStart.panX + dx;
      panY = dragStart.panY + dy;
      clampPan();
      if (tooltipEl) tooltipEl.hidden = true;
      return;
    }

    hovered = hitTest(pt.x, pt.y);
    canvas.style.cursor = loupeEnabled
      ? 'none'
      : hovered
        ? 'pointer'
        : zoom > 1.01
          ? 'grab'
          : 'default';
    updateTooltip(pt);
  });

  canvas.addEventListener('pointerleave', () => {
    pointer.inside = false;
    hovered = null;
    if (tooltipEl) tooltipEl.hidden = true;
  });

  canvas.addEventListener('click', (e) => {
    if (dragMoved) {
      dragMoved = false;
      return;
    }
    const pt = screenToWorld(e);
    const hit = hitTest(pt.x, pt.y);
    if (!hit) return;

    inspectedId = hit.id;
    const pal = getThemePalette();

    if (hit.kind === 'edge' && hit.layer === 2) {
      if (onSelectWeight) onSelectWeight(hit.id);
      const def = W2_DEFS.find((w) => w.key === hit.id);
      if (def) {
        triggerRipple(def.to, pal.back);
        triggerRipple(def.from, pal.fwd);
      }
    } else if (hit.kind === 'node') {
      triggerRipple(hit.id, hit.id.startsWith('o') ? pal.back : pal.fwd);
    }

    if (onInspectElement) {
      onInspectElement(hit.id);
    }
  });

  return {
    setState,
    setLoupe,
    setZoom,
    getZoom: () => zoom,
    triggerRipple,
    triggerWeightUpdateEffect,
    spawnFloatingText,
    destroy() {
      if (rafId) cancelAnimationFrame(rafId);
    },
  };
}
