/**
 * 10-은닉층(총 12층) 60fps 인터랙티브 줌/팬/미니맵/2.2× 돋보기 Canvas 엔진
 * - 1~10 은닉층(기본 10층 = 12층, 24노드, 44가중치, 3오차박스 = 71개 요소) 동적 레이아웃
 * - 커서 중심 마우스 휠 확대/축소, 클릭-드래그 패닝, 층별 스무스 카메라 포커스, 인터랙티브 미니맵
 * - 2.2× X-Ray 광학 돋보기(Z): 전체 보기 상태에서도 커서 주변을 확대해 내부 net/δ/기울기 투시
 * - 화이트 모드(기본) / 다크 모드 완벽 지원 + Blue(순전파) vs Orange(역전파) vs Emerald(오차 감소)
 */
import { activateDerivative, fmt, fmtSci } from '../math/deep-backprop.js';

const VIEW_W = 1600;
const VIEW_H = 540;
const LAYER_GAP = 216;
const NODE_R = 44;
const NODE_Y0 = 176;
const NODE_Y1 = 368;
const START_X = 96;

const MINIMAP = { x: VIEW_W - 306, y: VIEW_H - 76, w: 288, h: 58 };

function getThemePalette() {
  const isLight = document.documentElement.dataset.theme === 'light';
  if (isLight) {
    return {
      isLight: true,
      bgTop: '#f3f6fc',
      bgBot: '#e8eef7',
      orbFwd: 'rgba(37, 99, 235, 0.08)',
      orbBack: 'rgba(234, 88, 12, 0.08)',
      gridLine: 'rgba(15, 23, 42, 0.045)',
      heading: '#475569',
      textMain: '#0f172a',
      textSub: '#334155',
      textMuted: '#64748b',
      fwd: '#2563eb',
      fwdRgb: '37, 99, 235',
      fwdText: '#1d4ed8',
      fwdPillBg: 'rgba(239, 246, 255, 0.96)',
      back: '#ea580c',
      backRgb: '234, 88, 12',
      backText: '#c2410c',
      backPillBg: 'rgba(255, 247, 237, 0.96)',
      nodeFillTop: '#ffffff',
      nodeFillBot: '#edf2f9',
      nodeShadow: 'rgba(15, 23, 42, 0.12)',
      trackBg: 'rgba(15, 23, 42, 0.09)',
      pillBg: 'rgba(255, 255, 255, 0.94)',
      pillBorder: 'rgba(148, 163, 184, 0.45)',
      boxBg: 'rgba(255, 255, 255, 0.9)',
      boxFocusBg: 'rgba(255, 247, 237, 0.96)',
      boxBorder: 'rgba(148, 163, 184, 0.4)',
      minimapBg: 'rgba(255, 255, 255, 0.88)',
      loupeRing: '#1e293b',
      loupeShadow: 'rgba(15, 23, 42, 0.28)',
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
    fwdPillBg: 'rgba(23, 37, 84, 0.94)',
    back: '#f97316',
    backRgb: '249, 115, 22',
    backText: '#fb923c',
    backPillBg: 'rgba(67, 20, 7, 0.94)',
    nodeFillTop: '#24324d',
    nodeFillBot: '#111a2e',
    nodeShadow: 'rgba(0, 0, 0, 0.5)',
    trackBg: 'rgba(255, 255, 255, 0.1)',
    pillBg: 'rgba(24, 34, 54, 0.92)',
    pillBorder: 'rgba(148, 163, 184, 0.32)',
    boxBg: 'rgba(22, 31, 50, 0.88)',
    boxFocusBg: 'rgba(54, 23, 14, 0.94)',
    boxBorder: 'rgba(148, 163, 184, 0.28)',
    minimapBg: 'rgba(15, 23, 42, 0.86)',
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

/**
 * 기울기 크기를 0.08 ~ 1.0 시각 강도로 매핑 (기울기 소실이 극적으로 보이면서 ReLU는 밝게 유지)
 */
function gradVisualIntensity(meanAbsGrad, outMeanGrad) {
  if (!Number.isFinite(meanAbsGrad) || meanAbsGrad <= 0 || outMeanGrad <= 0) return 0.08;
  const ratio = meanAbsGrad / outMeanGrad;
  if (ratio >= 0.01) {
    // 1% ~ 100%+ 는 강하고 선명하게 (0.55 ~ 1.0)
    return Math.min(1.0, 0.55 + Math.pow(Math.min(1, ratio), 0.4) * 0.45);
  }
  // 1% 미만은 로그 감쇠로 뚜렷하게 어두워짐 (기울기 소실 직관화)
  const logVal = Math.log10(Math.max(1e-11, ratio)); // -11 .. -2
  return Math.max(0.08, 0.08 + ((logVal + 11) / 9) * 0.44);
}

export function createDeepNeuralCanvas(canvas, tooltipEl, { onInspectElement, onCameraChange }) {
  const ctx = canvas.getContext('2d');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  let currentBp = null;
  let currentWeights = null;
  let flowMode = 'backward'; // 'forward' | 'backward'
  let inspected = null; // { kind: 'node'|'edge'|'error', ... }
  let hovered = null;
  let loupeEnabled = false;

  // 동적 월드 레이아웃
  let layout = {
    numLayers: 12,
    worldW: 3320,
    worldH: VIEW_H,
    nodes: [],
    edges: [],
    errorBoxes: [],
    totalBox: null,
  };

  // 카메라 상태 (현재값 + 목표값 부드러운 보간)
  let fitScale = 0.47;
  const cam = {
    scale: 0.47,
    tx: 20,
    ty: 0,
    targetScale: 0.47,
    targetTx: 20,
    targetTy: 0,
  };

  let isDragging = false;
  let isDraggingMinimap = false;
  let dragMoved = false;
  let dragStart = { x: 0, y: 0, tx: 0, ty: 0 };
  let pointer = { vx: -999, vy: -999, wx: -999, wy: -999, inside: false };

  let pulseEffects = [];
  let floatingTexts = [];
  const startTime = performance.now();
  let rafId = null;

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(VIEW_W * dpr);
    canvas.height = Math.round(VIEW_H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function rebuildLayout(numHidden) {
    const numLayers = numHidden + 2;
    const nodes = [];
    const edges = [];

    for (let l = 0; l < numLayers; l++) {
      const x = START_X + l * LAYER_GAP;
      const layerTag = l === 0 ? 'In' : l === numLayers - 1 ? 'Out' : `H${l}`;
      for (let i = 0; i < 2; i++) {
        const y = i === 0 ? NODE_Y0 : NODE_Y1;
        const label =
          l === 0 ? `i${i + 1}` : l === numLayers - 1 ? `o${i + 1}` : `h${l},${i + 1}`;
        nodes.push({
          id: `node_${l}_${i}`,
          layer: l,
          idx: i,
          layerTag,
          label,
          x,
          y,
        });
      }
    }

    for (let m = 0; m < numLayers - 1; m++) {
      const fromX = START_X + m * LAYER_GAP;
      const toX = START_X + (m + 1) * LAYER_GAP;
      const fromTag = m === 0 ? 'In' : `H${m}`;
      const toTag = m === numLayers - 2 ? 'Out' : `H${m + 1}`;
      for (let r = 0; r < 2; r++) {
        for (let c = 0; c < 2; c++) {
          const y1 = r === 0 ? NODE_Y0 : NODE_Y1;
          const y2 = c === 0 ? NODE_Y0 : NODE_Y1;
          const t = r === c ? 0.5 : 0.31;
          edges.push({
            id: `edge_${m}_${r}_${c}`,
            m,
            r,
            c,
            fromId: `node_${m}_${r}`,
            toId: `node_${m + 1}_${c}`,
            fromLabel: m === 0 ? `i${r + 1}` : `h${m},${r + 1}`,
            toLabel: m === numLayers - 2 ? `o${c + 1}` : `h${m + 1},${c + 1}`,
            key: `W${m + 1}[${r + 1},${c + 1}]`,
            shortKey: `w${m + 1}_${r + 1}${c + 1}`,
            conn: `${fromTag}→${toTag}`,
            x1: fromX + NODE_R,
            y1,
            x2: toX - NODE_R,
            y2,
            lx: fromX + (toX - fromX) * t,
            ly: y1 + (y2 - y1) * t,
          });
        }
      }
    }

    const outX = START_X + (numLayers - 1) * LAYER_GAP;
    const boxX = outX + 96;
    const boxW = 290;
    const errorBoxes = [
      { id: 'E_o1', oKey: 'o1', idx: 0, x: boxX, y: 102, w: boxW, h: 136, cy: NODE_Y0 },
      { id: 'E_o2', oKey: 'o2', idx: 1, x: boxX, y: 294, w: boxW, h: 136, cy: NODE_Y1 },
    ];
    const totalBox = { id: 'E_total', x: boxX, y: 448, w: boxW, h: 46 };
    const worldW = Math.max(VIEW_W, boxX + boxW + 52);

    layout = { numLayers, worldW, worldH: VIEW_H, nodes, edges, errorBoxes, totalBox };
    fitScale = Math.min(1.0, (VIEW_W - 36) / worldW);
  }

  function clampCamera(scale, tx, ty) {
    const s = Math.max(fitScale * 0.92, Math.min(2.35, scale));
    const scaledW = layout.worldW * s;
    const scaledH = layout.worldH * s;
    let nx = tx;
    let ny = ty;

    if (scaledW <= VIEW_W) {
      nx = (VIEW_W - scaledW) / 2;
    } else {
      nx = Math.min(36, Math.max(VIEW_W - scaledW - 36, tx));
    }

    if (scaledH <= VIEW_H) {
      ny = (VIEW_H - scaledH) / 2;
    } else {
      ny = Math.min(24, Math.max(VIEW_H - scaledH - 24, ty));
    }
    return { scale: s, tx: nx, ty: ny };
  }

  function notifyCamera() {
    if (onCameraChange) {
      const relativePct = Math.round((cam.targetScale / fitScale) * 100);
      onCameraChange({ scale: cam.targetScale, fitScale, relativePct });
    }
  }

  function focusRegion(region, instant = false) {
    let targetScale = fitScale;
    let centerWorldX = layout.worldW * 0.5;
    let centerWorldY = layout.worldH * 0.5;

    if (region === 'fit') {
      targetScale = fitScale;
    } else if (region === 'front') {
      targetScale = Math.max(fitScale * 2.15, 0.96);
      centerWorldX = START_X + LAYER_GAP * 1.4;
    } else if (region === 'mid') {
      const midLayer = Math.max(2, Math.floor(layout.numLayers / 2));
      targetScale = Math.max(fitScale * 2.15, 0.96);
      centerWorldX = START_X + LAYER_GAP * midLayer;
    } else if (region === 'back') {
      targetScale = Math.max(fitScale * 2.05, 0.94);
      centerWorldX = layout.worldW - 640;
    } else if (typeof region === 'number') {
      const clampedLayer = Math.max(0, Math.min(layout.numLayers - 1, region));
      targetScale = Math.max(fitScale * 2.4, 1.08);
      centerWorldX = START_X + clampedLayer * LAYER_GAP;
    }

    const rawTx = VIEW_W / 2 - centerWorldX * targetScale;
    const rawTy = VIEW_H / 2 - centerWorldY * targetScale;
    const clamped = clampCamera(targetScale, rawTx, rawTy);
    cam.targetScale = clamped.scale;
    cam.targetTx = clamped.tx;
    cam.targetTy = clamped.ty;

    if (instant) {
      cam.scale = clamped.scale;
      cam.tx = clamped.tx;
      cam.ty = clamped.ty;
    }
    notifyCamera();
  }

  function zoomByFactor(factor, anchorVx = VIEW_W / 2, anchorVy = VIEW_H / 2) {
    const worldX = (anchorVx - cam.targetTx) / cam.targetScale;
    const worldY = (anchorVy - cam.targetTy) / cam.targetScale;
    const nextScale = Math.max(fitScale * 0.92, Math.min(2.35, cam.targetScale * factor));
    const nextTx = anchorVx - worldX * nextScale;
    const nextTy = anchorVy - worldY * nextScale;
    const clamped = clampCamera(nextScale, nextTx, nextTy);
    cam.targetScale = clamped.scale;
    cam.targetTx = clamped.tx;
    cam.targetTy = clamped.ty;
    notifyCamera();
  }

  function setState({ bp, weights, mode = 'backward', activeInspect, resetFit = false }) {
    const prevLayers = layout.numLayers;
    currentBp = bp;
    currentWeights = weights;
    flowMode = mode;
    if (activeInspect !== undefined) inspected = activeInspect;

    if (bp.before.numLayers !== prevLayers || resetFit) {
      rebuildLayout(bp.before.numHidden);
      focusRegion('fit', true);
    }
  }

  function triggerBackpropWave(errorReduction) {
    if (!currentBp) return;
    const pal = getThemePalette();
    const numLayers = layout.numLayers;
    const outMean = currentBp.layerStats[currentBp.layerStats.length - 1]?.meanAbsGrad || 1e-6;

    // 출력층부터 입력층까지 45ms 간격으로 순차 릴레이 파동!
    for (let step = 0; step < numLayers - 1; step++) {
      const layerIdx = numLayers - 1 - step;
      const stat = currentBp.layerStats[layerIdx - 1];
      const intensity = stat ? gradVisualIntensity(stat.meanAbsGrad, outMean) : 0.8;
      setTimeout(() => {
        const x = START_X + layerIdx * LAYER_GAP;
        pulseEffects.push({
          x,
          y: NODE_Y0,
          color: pal.back,
          radius: NODE_R,
          alpha: Math.min(1, 0.25 + intensity * 0.75),
        });
        pulseEffects.push({
          x,
          y: NODE_Y1,
          color: pal.back,
          radius: NODE_R,
          alpha: Math.min(1, 0.25 + intensity * 0.75),
        });
        if (stat && (layerIdx === numLayers - 1 || layerIdx === 1 || layerIdx === Math.floor(numLayers / 2))) {
          floatingTexts.push({
            x: x - LAYER_GAP * 0.5,
            y: NODE_Y0 - 64,
            text: `${stat.label} Δ≈${fmtSci(stat.meanAbsDeltaW)}`,
            color: intensity > 0.35 ? pal.backText : pal.textMuted,
            alpha: 1.0,
            vy: -1.1,
          });
        }
      }, step * 45);
    }

    if (errorReduction > 0 && layout.totalBox) {
      floatingTexts.push({
        x: layout.totalBox.x + layout.totalBox.w / 2,
        y: layout.totalBox.y - 12,
        text: `오차 −${fmtSci(errorReduction)} 감소!`,
        color: '#10b981',
        alpha: 1.0,
        vy: -1.2,
      });
    }
  }

  function screenToCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const vx = ((e.clientX - rect.left) / rect.width) * VIEW_W;
    const vy = ((e.clientY - rect.top) / rect.height) * VIEW_H;
    const wx = (vx - cam.tx) / cam.scale;
    const wy = (vy - cam.ty) / cam.scale;
    return {
      vx,
      vy,
      wx,
      wy,
      clientX: e.clientX - rect.left,
      clientY: e.clientY - rect.top,
    };
  }

  function isInMinimap(vx, vy) {
    return (
      vx >= MINIMAP.x &&
      vx <= MINIMAP.x + MINIMAP.w &&
      vy >= MINIMAP.y &&
      vy <= MINIMAP.y + MINIMAP.h
    );
  }

  function panFromMinimap(vx, vy) {
    const ratioX = Math.max(0, Math.min(1, (vx - MINIMAP.x) / MINIMAP.w));
    const ratioY = Math.max(0, Math.min(1, (vy - MINIMAP.y) / MINIMAP.h));
    const worldX = ratioX * layout.worldW;
    const worldY = ratioY * layout.worldH;
    const rawTx = VIEW_W / 2 - worldX * cam.targetScale;
    const rawTy = VIEW_H / 2 - worldY * cam.targetScale;
    const clamped = clampCamera(cam.targetScale, rawTx, rawTy);
    cam.targetTx = clamped.tx;
    cam.targetTy = clamped.ty;
  }

  function hitTest(wx, wy) {
    for (const n of layout.nodes) {
      if (Math.hypot(wx - n.x, wy - n.y) <= NODE_R + 6) {
        return { kind: 'node', id: n.id, ref: n };
      }
    }
    for (const b of layout.errorBoxes) {
      if (wx >= b.x && wx <= b.x + b.w && wy >= b.y && wy <= b.y + b.h) {
        return { kind: 'error', id: b.id, ref: b };
      }
    }
    if (
      layout.totalBox &&
      wx >= layout.totalBox.x &&
      wx <= layout.totalBox.x + layout.totalBox.w &&
      wy >= layout.totalBox.y &&
      wy <= layout.totalBox.y + layout.totalBox.h
    ) {
      return { kind: 'error', id: 'E_total', ref: layout.totalBox };
    }
    for (const ed of layout.edges) {
      if (Math.abs(wx - ed.lx) <= 50 && Math.abs(wy - ed.ly) <= 20) {
        return { kind: 'edge', id: ed.id, ref: ed };
      }
      if (pointToSegmentDist(wx, wy, ed.x1, ed.y1, ed.x2, ed.y2) <= 14) {
        return { kind: 'edge', id: ed.id, ref: ed };
      }
    }
    return null;
  }

  function isRelatedToHover(kind, obj) {
    if (!hovered) return false;
    if (hovered.kind === 'node') {
      if (kind === 'node') return obj.id === hovered.id;
      if (kind === 'edge') return obj.fromId === hovered.id || obj.toId === hovered.id;
    }
    if (hovered.kind === 'edge') {
      if (kind === 'edge') return obj.id === hovered.id;
      if (kind === 'node') return obj.id === hovered.ref.fromId || obj.id === hovered.ref.toId;
    }
    if (hovered.kind === 'error') {
      if (kind === 'error') return obj.id === hovered.id || hovered.id === 'E_total';
    }
    return false;
  }

  function updateTooltip(pt) {
    if (!tooltipEl || !currentBp || loupeEnabled || isDragging || isDraggingMinimap || !hovered) {
      if (tooltipEl) tooltipEl.hidden = true;
      return;
    }
    const { before, deltas, grads, layerStats } = currentBp;
    let html = '';

    if (hovered.kind === 'node') {
      const n = hovered.ref;
      const actVal = before.acts[n.layer][n.idx];
      const netVal = before.nets[n.layer]?.[n.idx];
      const deltaVal = deltas[n.layer]?.[n.idx];
      html = `
        <div class="tt-title">${n.layerTag} · ${n.label} <span class="tt-hint">클릭해 층별 해부</span></div>
        <div>출력값 <strong class="num">${fmt(actVal, 6)}</strong>${netVal != null ? ` · 가중합 <strong class="num">${fmt(netVal, 4)}</strong>` : ''}</div>
        ${deltaVal != null ? `<div>역전파 델타(δ) <strong class="num text-back">${fmtSci(deltaVal)}</strong></div>` : '<div>입력층 원본 신호</div>'}
      `;
    } else if (hovered.kind === 'edge') {
      const ed = hovered.ref;
      const wVal = currentWeights[ed.m][ed.r][ed.c];
      const gradVal = grads[ed.m][ed.r][ed.c];
      const stat = layerStats[ed.m];
      html = `
        <div class="tt-title">${ed.key} (${ed.fromLabel} → ${ed.toLabel}) <span class="tt-hint">클릭해 상세 해설</span></div>
        <div>가중치 <strong class="num">${fmt(wVal, 4)}</strong> · 기울기 <strong class="num text-back">${fmtSci(gradVal)}</strong></div>
        <div>출력층 대비 기울기 세기: <strong class="num">${(stat.relativeToOut * 100).toFixed(4)}%</strong></div>
      `;
    } else if (hovered.kind === 'error') {
      const val =
        hovered.id === 'E_o1'
          ? fmt(before.errors[0], 6)
          : hovered.id === 'E_o2'
            ? fmt(before.errors[1], 6)
            : fmt(before.eTotal, 6);
      html = `
        <div class="tt-title">${hovered.id} = <span class="num">${val}</span> <span class="tt-hint">클릭해 공식 보기</span></div>
        <div>전체 ${before.numHidden}개 은닉층이 이 오차를 줄이기 위해 역전파로 수정됩니다.</div>
      `;
    }

    tooltipEl.hidden = false;
    tooltipEl.innerHTML = html;
    tooltipEl.style.left = `${Math.min(Math.max(12, pt.clientX + 16), canvas.clientWidth - 310)}px`;
    tooltipEl.style.top = `${Math.max(12, pt.clientY - 82)}px`;
  }

  function drawStreamParticles(x1, y1, x2, y2, count, speed, colorRgb, radius, reverse, timeSec, alphaScale, pal) {
    if (alphaScale < 0.09) return;
    for (let i = 0; i < count; i++) {
      const prog = (timeSec * speed + i / count) % 1;
      const t = reverse ? 1 - prog : prog;
      const px = x1 + (x2 - x1) * t;
      const py = y1 + (y2 - y1) * t;

      const tailT = reverse ? Math.min(1, t + 0.1) : Math.max(0, t - 0.1);
      const tx = x1 + (x2 - x1) * tailT;
      const ty = y1 + (y2 - y1) * tailT;

      const grad = ctx.createLinearGradient(tx, ty, px, py);
      grad.addColorStop(0, `rgba(${colorRgb}, 0)`);
      grad.addColorStop(1, `rgba(${colorRgb}, ${(0.95 * alphaScale).toFixed(2)})`);

      ctx.strokeStyle = grad;
      ctx.lineWidth = radius * 1.8;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(px, py);
      ctx.stroke();

      ctx.fillStyle = pal.isLight ? `rgba(${colorRgb}, ${alphaScale.toFixed(2)})` : `rgba(255,255,255,${alphaScale.toFixed(2)})`;
      ctx.beginPath();
      ctx.arc(px, py, radius * 0.85, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function renderWorld(timeSec, pal, detailLevel) {
    // 월드 그리드
    ctx.strokeStyle = pal.gridLine;
    ctx.lineWidth = 1 / Math.max(0.5, cam.scale);
    const step = 60;
    ctx.beginPath();
    for (let gx = 0; gx <= layout.worldW; gx += step) {
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, layout.worldH);
    }
    for (let gy = 0; gy <= layout.worldH; gy += step) {
      ctx.moveTo(0, gy);
      ctx.lineTo(layout.worldW, gy);
    }
    ctx.stroke();

    if (!currentBp) return;
    const { before, deltas, grads, layerStats } = currentBp;
    const isBack = flowMode === 'backward';
    const outMean = layerStats[layerStats.length - 1]?.meanAbsGrad || 1e-6;

    // 1) 층별 상단 헤더 & 역전파 기울기 강도 배지
    for (let l = 0; l < layout.numLayers; l++) {
      const x = START_X + l * LAYER_GAP;
      const tag = l === 0 ? '입력 (In)' : l === layout.numLayers - 1 ? '출력 (Out)' : `은닉 H${l}`;
      ctx.font = '800 17px "Pretendard", sans-serif';
      ctx.fillStyle = pal.heading;
      ctx.textAlign = 'center';
      ctx.fillText(tag, x, 46);

      if (l >= 1) {
        const stat = layerStats[l - 1];
        const intensity = gradVisualIntensity(stat.meanAbsGrad, outMean);
        const barW = 90;
        drawRoundedRect(ctx, x - barW / 2, 58, barW, 8, 4);
        ctx.fillStyle = pal.trackBg;
        ctx.fill();
        drawRoundedRect(ctx, x - barW / 2, 58, Math.max(4, barW * intensity), 8, 4);
        ctx.fillStyle = isBack ? pal.back : pal.fwd;
        ctx.fill();

        if (detailLevel >= 1) {
          ctx.font = '700 12px ui-monospace, SFMono-Regular, monospace';
          ctx.fillStyle = isBack ? pal.backText : pal.textMuted;
          ctx.fillText(`|∇| ${fmtSci(stat.meanAbsGrad)}`, x, 84);
        }
      }
    }

    // 2) 연결선 44개 렌더링
    for (const ed of layout.edges) {
      const wVal = currentWeights[ed.m][ed.r][ed.c];
      const gradVal = grads[ed.m][ed.r][ed.c];
      const stat = layerStats[ed.m];
      const intensity = gradVisualIntensity(stat.meanAbsGrad, outMean);
      const hovRel = isRelatedToHover('edge', ed);
      const isInsp = inspected?.id === ed.id;

      ctx.save();
      const baseW = 2.0 + Math.min(4.2, Math.abs(wVal) * 4.5);
      if (isBack) {
        const alpha = hovRel || isInsp ? 1.0 : Math.max(0.14, intensity);
        ctx.strokeStyle = `rgba(${pal.backRgb}, ${alpha.toFixed(2)})`;
        ctx.lineWidth = baseW * (0.6 + intensity * 0.9) + (hovRel || isInsp ? 2.4 : 0);
        if (intensity < 0.2 && !hovRel && !isInsp) {
          ctx.setLineDash([6, 6]); // 기울기 소실 층은 점선으로 시각 경고!
        }
      } else {
        ctx.strokeStyle = `rgba(${pal.fwdRgb}, ${hovRel || isInsp ? 1.0 : 0.72})`;
        ctx.lineWidth = baseW + (hovRel || isInsp ? 2.2 : 0);
      }

      ctx.beginPath();
      ctx.moveTo(ed.x1, ed.y1);
      ctx.lineTo(ed.x2, ed.y2);
      ctx.stroke();
      ctx.setLineDash([]);

      // 입자 흐름 (역전파 시 실제 기울기 세기에 비례해 입자 수·속도·밝기 변화!)
      if (isBack) {
        const pCount = intensity > 0.6 ? 3 : intensity > 0.25 ? 2 : 1;
        const pSpeed = 0.16 + intensity * 0.52;
        drawStreamParticles(ed.x1, ed.y1, ed.x2, ed.y2, pCount, pSpeed, pal.backRgb, 2.2 + intensity * 2.6, true, timeSec, intensity, pal);
      } else {
        drawStreamParticles(ed.x1, ed.y1, ed.x2, ed.y2, 2, 0.34, pal.fwdRgb, 3.4, false, timeSec, 0.9, pal);
      }

      // 확대 시 또는 돋보기 X-Ray 시 가중치 배지 표시 (전체 축소 보기에서는 화면 복잡도 방지)
      if (detailLevel >= 1 || hovRel || isInsp) {
        const bw = detailLevel >= 2 ? 104 : 84;
        const bh = detailLevel >= 2 ? 36 : 24;
        drawRoundedRect(ctx, ed.lx - bw / 2, ed.ly - bh / 2, bw, bh, 7);
        ctx.fillStyle = hovRel || isInsp ? (isBack ? pal.backPillBg : pal.fwdPillBg) : pal.pillBg;
        ctx.fill();
        ctx.strokeStyle = hovRel || isInsp ? (isBack ? pal.back : pal.fwd) : pal.pillBorder;
        ctx.lineWidth = hovRel || isInsp ? 2 : 1.1;
        ctx.stroke();

        ctx.font = '700 12.5px ui-monospace, SFMono-Regular, monospace';
        ctx.fillStyle = hovRel || isInsp ? (isBack ? pal.backText : pal.fwdText) : pal.textMain;
        ctx.textAlign = 'center';
        ctx.fillText(fmt(wVal, 2), ed.lx, detailLevel >= 2 ? ed.ly - 2 : ed.ly + 4);

        if (detailLevel >= 2) {
          ctx.font = '600 10px ui-monospace, SFMono-Regular, monospace';
          ctx.fillStyle = isBack ? pal.backText : pal.fwdText;
          ctx.fillText(`∇${fmtSci(gradVal)}`, ed.lx, ed.ly + 12);
        }
      }
      ctx.restore();
    }

    // 3) 오차 박스 렌더링
    const outX = START_X + (layout.numLayers - 1) * LAYER_GAP;
    for (const box of layout.errorBoxes) {
      const target = before.targets[box.idx];
      const out = before.out[box.idx];
      const err = before.errors[box.idx];
      const hovRel = isRelatedToHover('error', box);
      const isInsp = inspected?.id === box.id;

      ctx.save();
      ctx.strokeStyle = pal.back;
      ctx.lineWidth = hovRel || isInsp ? 4 : 3;
      ctx.beginPath();
      ctx.moveTo(outX + NODE_R, box.cy);
      ctx.lineTo(box.x - 4, box.cy);
      ctx.stroke();

      drawStreamParticles(outX + NODE_R + 4, box.cy, box.x - 6, box.cy, 4, 0.68, pal.backRgb, 4.4, isBack, timeSec, 1.0, pal);

      drawRoundedRect(ctx, box.x, box.y, box.w, box.h, 14);
      ctx.fillStyle = isBack || hovRel || isInsp ? pal.boxFocusBg : pal.boxBg;
      ctx.fill();
      ctx.strokeStyle = isBack || hovRel || isInsp ? pal.back : pal.boxBorder;
      ctx.lineWidth = hovRel || isInsp ? 2.6 : 1.8;
      ctx.stroke();

      ctx.textAlign = 'left';
      ctx.font = '700 16px "Pretendard", sans-serif';
      ctx.fillStyle = pal.textSub;
      ctx.fillText(`목표 ${box.oKey}`, box.x + 18, box.y + 34);
      ctx.fillText(`출력 ${box.oKey}`, box.x + 18, box.y + 64);

      ctx.textAlign = 'right';
      ctx.font = '800 17px ui-monospace, SFMono-Regular, monospace';
      ctx.fillStyle = pal.textMain;
      ctx.fillText(fmt(target, 2), box.x + box.w - 18, box.y + 34);
      ctx.fillStyle = isBack ? pal.backText : pal.fwdText;
      ctx.fillText(fmt(out, 4), box.x + box.w - 18, box.y + 64);

      // 오차 바
      const bx = box.x + 18;
      const by = box.y + 80;
      const bw = box.w - 36;
      drawRoundedRect(ctx, bx, by, bw, 9, 4.5);
      ctx.fillStyle = pal.trackBg;
      ctx.fill();
      drawRoundedRect(ctx, bx + Math.min(target, out) * bw, by, Math.max(0.02, Math.abs(target - out)) * bw, 9, 4.5);
      ctx.fillStyle = pal.back;
      ctx.fill();

      ctx.textAlign = 'left';
      ctx.font = '800 16px ui-monospace, SFMono-Regular, monospace';
      ctx.fillStyle = pal.backText;
      ctx.fillText(`E_${box.oKey} = ${fmt(err, 4)}`, box.x + 18, box.y + 118);
      ctx.restore();
    }

    // 전체 오차 카드
    if (layout.totalBox) {
      const tb = layout.totalBox;
      const hovTotal = isRelatedToHover('error', tb) || inspected?.id === 'E_total';
      ctx.save();
      drawRoundedRect(ctx, tb.x, tb.y, tb.w, tb.h, 12);
      ctx.fillStyle = pal.boxFocusBg;
      ctx.fill();
      ctx.strokeStyle = hovTotal ? pal.fwd : pal.back;
      ctx.lineWidth = hovTotal ? 2.6 : 2;
      ctx.stroke();
      ctx.textAlign = 'center';
      ctx.font = '800 17.5px ui-monospace, SFMono-Regular, monospace';
      ctx.fillStyle = pal.backText;
      ctx.fillText(`전체 오차 E_total = ${fmt(before.eTotal, 6)}`, tb.x + tb.w / 2, tb.y + 29);
      ctx.restore();
    }

    // 4) 충격파 & 플로팅 텍스트
    if (detailLevel < 3) {
      pulseEffects = pulseEffects.filter((p) => p.alpha > 0.03);
      for (const p of pulseEffects) {
        ctx.save();
        ctx.strokeStyle = p.color;
        ctx.globalAlpha = p.alpha;
        ctx.lineWidth = 3.5;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
        p.radius += 3.2;
        p.alpha *= 0.9;
      }

      floatingTexts = floatingTexts.filter((ft) => ft.alpha > 0.03);
      for (const ft of floatingTexts) {
        ctx.save();
        ctx.globalAlpha = ft.alpha;
        ctx.font = '900 18px ui-monospace, SFMono-Regular, monospace';
        ctx.fillStyle = ft.color;
        ctx.textAlign = 'center';
        ctx.fillText(ft.text, ft.x, ft.y);
        ctx.restore();
        ft.y += ft.vy;
        ft.alpha *= 0.965;
      }
    }

    // 5) 24개 뉴런 노드 렌더링
    for (const n of layout.nodes) {
      const actVal = before.acts[n.layer][n.idx];
      const netVal = before.nets[n.layer]?.[n.idx];
      const deltaVal = deltas[n.layer]?.[n.idx];
      const hovRel = isRelatedToHover('node', n);
      const isInsp = inspected?.id === n.id;

      const stat = n.layer >= 1 ? layerStats[n.layer - 1] : null;
      const intensity = stat ? gradVisualIntensity(stat.meanAbsGrad, outMean) : 0.6;
      const focusColor = isBack && n.layer >= 1 ? pal.back : pal.fwd;
      const focusText = isBack && n.layer >= 1 ? pal.backText : pal.fwdText;

      ctx.save();
      if (isBack && n.layer >= 1 && !hovRel && !isInsp) {
        ctx.globalAlpha = Math.max(0.42, intensity);
      }

      if (hovRel || isInsp) {
        ctx.strokeStyle = focusColor;
        ctx.lineWidth = 3;
        ctx.setLineDash([6, 5]);
        ctx.beginPath();
        ctx.arc(n.x, n.y, NODE_R + 10, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      const r = hovRel || isInsp ? NODE_R + 3 : NODE_R;
      ctx.shadowColor = hovRel || isInsp ? focusColor : pal.nodeShadow;
      ctx.shadowBlur = hovRel || isInsp ? 20 : 12;
      const g = ctx.createRadialGradient(n.x - 14, n.y - 14, 4, n.x, n.y, r + 4);
      g.addColorStop(0, pal.nodeFillTop);
      g.addColorStop(1, pal.nodeFillBot);
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.strokeStyle = pal.pillBorder;
      ctx.lineWidth = 3;
      ctx.stroke();

      // 원형 활성도 아크
      const ratio = Math.max(0.05, Math.min(1, Math.abs(actVal)));
      ctx.strokeStyle = focusColor;
      ctx.lineWidth = 4.6;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.arc(n.x, n.y, r, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * ratio);
      ctx.stroke();

      ctx.textAlign = 'center';
      ctx.font = '800 18px "Pretendard", sans-serif';
      ctx.fillStyle = pal.textMain;
      ctx.fillText(n.label, n.x, n.y - 5);

      ctx.font = '800 15.5px ui-monospace, SFMono-Regular, monospace';
      ctx.fillStyle = focusText;
      ctx.fillText(fmt(actVal, 3), n.x, n.y + 17);

      // 확대 / 돋보기 X-Ray 시 노드 하단에 델타·기울기 상세 표시
      if (detailLevel >= 2 && n.layer >= 1) {
        const layerKind = n.layer === layout.numLayers - 1 ? 'sigmoid' : before.activation;
        const slope = activateDerivative(netVal, actVal, layerKind);
        ctx.font = '700 12px ui-monospace, SFMono-Regular, monospace';
        ctx.fillStyle = isBack ? pal.backText : pal.textSub;
        ctx.fillText(isBack ? `δ=${fmtSci(deltaVal)}` : `net=${fmt(netVal, 2)} · σ'=${fmt(slope, 2)}`, n.x, n.y + r + 22);
      }
      ctx.restore();
    }
  }

  /**
   * 우측 하단 인터랙티브 미니맵 (12층 전체 조망 + 뷰포트 사각형)
   */
  function drawMinimap(pal) {
    if (!currentBp) return;
    const { x, y, w, h } = MINIMAP;
    const sx = w / layout.worldW;
    const sy = h / layout.worldH;
    const outMean = currentBp.layerStats[currentBp.layerStats.length - 1]?.meanAbsGrad || 1e-6;

    ctx.save();
    drawRoundedRect(ctx, x, y, w, h, 10);
    ctx.fillStyle = pal.minimapBg;
    ctx.fill();
    ctx.strokeStyle = pal.boxBorder;
    ctx.lineWidth = 1.4;
    ctx.stroke();

    // 각 층 노드 점 + 기울기 강도 색
    for (const n of layout.nodes) {
      const stat = n.layer >= 1 ? currentBp.layerStats[n.layer - 1] : null;
      const intensity = stat ? gradVisualIntensity(stat.meanAbsGrad, outMean) : 0.7;
      ctx.fillStyle =
        flowMode === 'backward' && n.layer >= 1
          ? `rgba(${pal.backRgb}, ${Math.max(0.2, intensity).toFixed(2)})`
          : pal.fwd;
      ctx.beginPath();
      ctx.arc(x + n.x * sx, y + n.y * sy, 3.2, 0, Math.PI * 2);
      ctx.fill();
    }

    // 현재 카메라 뷰포트 사각형
    const vx0 = Math.max(0, -cam.tx / cam.scale);
    const vy0 = Math.max(0, -cam.ty / cam.scale);
    const vw = Math.min(layout.worldW, VIEW_W / cam.scale);
    const vh = Math.min(layout.worldH, VIEW_H / cam.scale);

    drawRoundedRect(ctx, x + vx0 * sx, y + vy0 * sy, Math.max(16, vw * sx), Math.max(12, vh * sy), 5);
    ctx.fillStyle = `rgba(${pal.fwdRgb}, 0.14)`;
    ctx.fill();
    ctx.strokeStyle = pal.fwd;
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.font = '700 10.5px "Pretendard", sans-serif';
    ctx.fillStyle = pal.textMuted;
    ctx.textAlign = 'left';
    ctx.fillText('미니맵 (클릭·드래그 이동)', x + 8, y - 6);
    ctx.restore();
  }

  /**
   * 2.2× 광학 돋보기(Loupe) — 전체 보기에서도 커서 주변을 2.4배 확대 + X-Ray 내부값 투시!
   */
  function drawLoupe(timeSec, pal) {
    if (!loupeEnabled || !pointer.inside || isInMinimap(pointer.vx, pointer.vy)) return;
    const r = 132;
    const mag = Math.max(1.25, cam.scale * 2.35);
    const lx = pointer.vx;
    const ly = pointer.vy;

    ctx.save();
    ctx.shadowColor = pal.loupeShadow;
    ctx.shadowBlur = 28;
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
    ctx.translate(-pointer.wx, -pointer.wy);
    renderWorld(timeSec, pal, 3); // detailLevel 3 = Full X-Ray!
    ctx.restore();

    ctx.save();
    ctx.strokeStyle = pal.loupeRing;
    ctx.lineWidth = 3.2;
    ctx.beginPath();
    ctx.arc(lx, ly, r, 0, Math.PI * 2);
    ctx.stroke();

    drawRoundedRect(ctx, lx - 54, ly + r - 15, 108, 26, 13);
    ctx.fillStyle = pal.fwd;
    ctx.fill();
    ctx.font = '800 12px ui-monospace, SFMono-Regular, monospace';
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.fillText('X-RAY 돋보기', lx, ly + r + 2);
    ctx.restore();
  }

  function draw(now) {
    const timeSec = reducedMotion ? 0.35 : (now - startTime) / 1000;
    const pal = getThemePalette();

    // 부드러운 카메라 Lerp 보간
    const lerp = reducedMotion ? 1 : 0.16;
    cam.scale += (cam.targetScale - cam.scale) * lerp;
    cam.tx += (cam.targetTx - cam.tx) * lerp;
    cam.ty += (cam.targetTy - cam.ty) * lerp;

    // 캔버스 배경
    const bgGrad = ctx.createLinearGradient(0, 0, VIEW_W, VIEW_H);
    bgGrad.addColorStop(0, pal.bgTop);
    bgGrad.addColorStop(1, pal.bgBot);
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, VIEW_W, VIEW_H);

    // 월드 변환 렌더링
    ctx.save();
    ctx.translate(cam.tx, cam.ty);
    ctx.scale(cam.scale, cam.scale);
    const detailLevel = cam.scale >= 0.95 ? 2 : cam.scale >= 0.68 ? 1 : 0;
    renderWorld(timeSec, pal, detailLevel);
    ctx.restore();

    // 좌하단 안내 HUD
    ctx.save();
    ctx.font = '700 14px "Pretendard", sans-serif';
    ctx.fillStyle = flowMode === 'backward' ? pal.backText : pal.fwdText;
    ctx.textAlign = 'left';
    const actLabel = currentBp?.before.activation === 'relu' ? 'ReLU' : 'Sigmoid';
    const hudText =
      flowMode === 'backward'
        ? `역전파 (← 주황 · ${actLabel}): 마우스 휠로 원하는 층을 확대하거나 드래그해 10개 은닉층을 탐험하세요`
        : `순전파 (파랑 → · ${actLabel}): 입력 신호가 ${currentBp?.before.numHidden ?? 10}개 은닉층을 관통해 출력층으로 흐릅니다`;
    ctx.fillText(hudText, 26, VIEW_H - 18);
    ctx.restore();

    drawMinimap(pal);
    drawLoupe(timeSec, pal);

    rafId = requestAnimationFrame(draw);
  }

  rebuildLayout(10);
  resize();
  focusRegion('fit', true);
  rafId = requestAnimationFrame(draw);

  // ===== 인터랙션: 휠 줌 · 드래그 팬 · 미니맵 점프 · 호버 · 클릭 =====
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      const pt = screenToCoords(e);
      const factor = e.deltaY < 0 ? 1.16 : 1 / 1.16;
      zoomByFactor(factor, pt.vx, pt.vy);
    },
    { passive: false }
  );

  canvas.addEventListener('pointerdown', (e) => {
    const pt = screenToCoords(e);
    if (isInMinimap(pt.vx, pt.vy)) {
      isDraggingMinimap = true;
      panFromMinimap(pt.vx, pt.vy);
      return;
    }
    isDragging = true;
    dragMoved = false;
    dragStart = { x: pt.vx, y: pt.vy, tx: cam.targetTx, ty: cam.targetTy };
    canvas.style.cursor = 'grabbing';
  });

  window.addEventListener('pointerup', () => {
    isDragging = false;
    isDraggingMinimap = false;
  });

  canvas.addEventListener('pointermove', (e) => {
    const pt = screenToCoords(e);
    pointer = { vx: pt.vx, vy: pt.vy, wx: pt.wx, wy: pt.wy, inside: true };

    if (isDraggingMinimap) {
      panFromMinimap(pt.vx, pt.vy);
      if (tooltipEl) tooltipEl.hidden = true;
      return;
    }

    if (isDragging) {
      const dx = pt.vx - dragStart.x;
      const dy = pt.vy - dragStart.y;
      if (Math.hypot(dx, dy) > 4) dragMoved = true;
      const clamped = clampCamera(cam.targetScale, dragStart.tx + dx, dragStart.ty + dy);
      cam.targetTx = clamped.tx;
      cam.targetTy = clamped.ty;
      if (tooltipEl) tooltipEl.hidden = true;
      return;
    }

    if (isInMinimap(pt.vx, pt.vy)) {
      hovered = null;
      canvas.style.cursor = 'pointer';
      if (tooltipEl) tooltipEl.hidden = true;
      return;
    }

    hovered = hitTest(pt.wx, pt.wy);
    canvas.style.cursor = loupeEnabled ? 'none' : hovered ? 'pointer' : 'grab';
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
    const pt = screenToCoords(e);
    if (isInMinimap(pt.vx, pt.vy)) return;
    const hit = hitTest(pt.wx, pt.wy);
    if (!hit) return;

    inspected = hit;
    if (onInspectElement) onInspectElement(hit);
  });

  return {
    setState,
    focusRegion,
    zoomByFactor,
    setLoupe(enabled) {
      loupeEnabled = Boolean(enabled);
      if (loupeEnabled && tooltipEl) tooltipEl.hidden = true;
    },
    triggerBackpropWave,
    destroy() {
      if (rafId) cancelAnimationFrame(rafId);
    },
  };
}
