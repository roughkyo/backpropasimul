/**
 * 은닉층 10개(총 12층) 딥러닝 역전파 확장 시뮬레이터 모듈
 * - 1~10개 은닉층 실시간 전환 (1층 = 기존 M5 노트북 기준값과 100% 일치, 기본 10층 = 12층·24노드·44가중치)
 * - 커서 중심 줌/팬/미니맵/2.2× X-Ray 돋보기 + 층별 스무스 카메라 점프
 * - Sigmoid(기울기 소실) vs ReLU(신호 보존) 원클릭 비교 + 층별 기울기 감쇠 이퀄라이저 차트
 */
import {
  DEEP_DEFAULT_CONFIG,
  activateDerivative,
  cloneWeights,
  createInitialWeights,
  deepBackprop,
  fmt,
  fmtSci,
} from '../math/deep-backprop.js';
import { createDeepNeuralCanvas } from '../visuals/deep-neural-canvas.js';
import { mountEntropyCanvas } from '../visuals/entropy-bg.js';
import { flashElement, showToast } from '../core/ui.js';

const DEEP_KEYWORDS = [
  {
    id: 'kw_deep10',
    label: '#10은닉층(총12층)',
    sub: '24노드 · 44가중치 연쇄',
    action: 'overview',
  },
  {
    id: 'kw_vanish',
    label: '#기울기소실(Sigmoid)',
    sub: '0.25를 10번 곱하면 소멸',
    action: 'vanish',
  },
  {
    id: 'kw_relu',
    label: '#ReLU신호보존',
    sub: '기울기 1 유지 · 앞층 부활',
    action: 'relu',
  },
  {
    id: 'kw_depth_cmp',
    label: '#깊이비교(1층vs10층)',
    sub: '층이 깊어질수록 생기는 변화',
    action: 'compare',
  },
];

function buildMiniActivationSvg(kind, netVal, actVal, slope) {
  const isRelu = kind === 'relu';
  const clampedNet = Math.max(-4, Math.min(4, netVal));
  const px = 24 + ((clampedNet + 4) / 8) * 232;
  const normY = isRelu
    ? Math.max(0, Math.min(1, (actVal + 0.3) / 2.2))
    : Math.max(0, Math.min(1, actVal));
  const py = 70 - normY * 54;

  const pts = [];
  for (let i = 0; i <= 40; i++) {
    const x = -4 + (i / 40) * 8;
    const yVal = isRelu ? (x > 0 ? x : 0.05 * x) : 1 / (1 + Math.exp(-x));
    const ny = isRelu
      ? Math.max(0, Math.min(1, (yVal + 0.3) / 2.2))
      : Math.max(0, Math.min(1, yVal));
    const sx = 24 + (i / 40) * 232;
    const sy = 70 - ny * 54;
    pts.push(`${sx.toFixed(1)},${sy.toFixed(1)}`);
  }

  return `
    <svg class="insp-svg" viewBox="0 0 280 84" aria-hidden="true">
      <line x1="24" y1="70" x2="256" y2="70" stroke="var(--border-strong)" stroke-width="1" />
      <line x1="140" y1="12" x2="140" y2="74" stroke="var(--border)" stroke-dasharray="3 3" stroke-width="1" />
      <polyline fill="none" stroke="${isRelu ? 'var(--success)' : 'var(--fwd-solid)'}" stroke-width="2.6" points="${pts.join(' ')}" />
      <circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="5.5" fill="var(--back-solid)" stroke="var(--surface-solid)" stroke-width="2" />
      <text x="26" y="20" fill="var(--text-dim)" font-size="10" font-weight="700">${isRelu ? 'Leaky ReLU 곡선' : 'Sigmoid 곡선'}</text>
      <text x="254" y="20" text-anchor="end" fill="var(--back)" font-size="10.5" font-family="var(--font-mono)" font-weight="800">기울기 σ'=${fmt(slope, 3)}</text>
    </svg>
  `;
}

function buildDeepInspectorContent(hit, bp, weights) {
  const { before, deltas, grads, nextWeights, layerStats, lr } = bp;

  if (hit.kind === 'node') {
    const n = hit.ref;
    const actVal = before.acts[n.layer][n.idx];
    const netVal = before.nets[n.layer]?.[n.idx];
    const deltaVal = deltas[n.layer]?.[n.idx];
    const isInput = n.layer === 0;
    const isOutput = n.layer === before.numLayers - 1;
    const layerKind = isOutput ? 'sigmoid' : before.activation;

    if (isInput) {
      return {
        badge: '입력층 · 원본 신호',
        badgeClass: 'is-fwd',
        title: `${n.label} (입력 노드 ${n.idx + 1})`,
        bodyHtml: `
          <p class="insp-lead">네트워크의 출발점입니다. 이 값이 <strong>W1</strong>을 거쳐 첫 번째 은닉층 <strong>H1</strong>로 들어갑니다.</p>
          <div class="insp-calc">
            <div class="insp-calc-row"><span>현재 입력값</span><strong class="num text-fwd">${fmt(actVal, 4)}</strong></div>
            <div class="insp-calc-row"><span>연결된 다음 층</span><strong>은닉층 H1 (가중치 W1)</strong></div>
          </div>
          <p class="insp-tip">아래 슬라이더에서 입력값을 바꾸면 10개 은닉층을 지나 출력층까지 연쇄 변화합니다.</p>
        `,
      };
    }

    const slope = activateDerivative(netVal, actVal, layerKind);
    const outStat = layerStats[layerStats.length - 1];
    const myStat = layerStats[n.layer - 1];
    const pct = outStat && myStat ? (myStat.relativeToOut * 100).toFixed(4) : '100.0000';
    const vanishWarn = !isOutput && myStat && myStat.relativeToOut < 0.005;

    return {
      badge: isOutput ? '출력층 · 시그모이드' : `은닉층 ${n.layerTag} · ${before.activation === 'relu' ? 'ReLU' : 'Sigmoid'}`,
      badgeClass: 'is-back',
      title: `${n.label} (${n.layerTag} · ${n.idx + 1}번 뉴런)`,
      bodyHtml: `
        <p class="insp-lead">
          ${
            isOutput
              ? `최종 예측값 <strong>${fmt(actVal, 4)}</strong>을 목표값(<strong>${fmt(before.targets[n.idx], 2)}</strong>)과 비교해 첫 역전파 신호(δ)를 만듭니다.`
              : vanishWarn
                ? `뒤층에서 넘어오는 동안 <strong>시그모이드 기울기(≤0.25)</strong>가 반복 곱해져 오차 신호가 출력층의 <strong>${pct}%</strong>까지 작아졌습니다.`
                : `뒤층에서 넘어온 오차 신호에 활성화 기울기(<strong>${fmt(slope, 3)}</strong>)를 곱해 앞층으로 전달합니다.`
          }
        </p>
        <div class="insp-calc">
          <div class="insp-calc-row"><span>가중합 (net)</span><strong class="num">${fmt(netVal, 5)}</strong></div>
          <div class="insp-calc-row"><span>출력값 (act)</span><strong class="num text-fwd">${fmt(actVal, 6)}</strong></div>
          <div class="insp-calc-row"><span>활성화 기울기 (σ')</span><strong class="num">${fmt(slope, 5)}</strong></div>
          <div class="insp-calc-row is-total"><span>역전파 델타 (δ)</span><strong class="num text-back">${fmtSci(deltaVal)}</strong></div>
        </div>
        ${buildMiniActivationSvg(layerKind, netVal, actVal, slope)}
        <p class="insp-tip">이 층(${myStat.label})의 평균 기울기 세기는 출력층 대비 <strong>${pct}%</strong>입니다.</p>
      `,
    };
  }

  if (hit.kind === 'edge') {
    const ed = hit.ref;
    const wOld = weights[ed.m][ed.r][ed.c];
    const wNew = nextWeights[ed.m][ed.r][ed.c];
    const grad = grads[ed.m][ed.r][ed.c];
    const deltaW = wNew - wOld;
    const prevAct = before.acts[ed.m][ed.r];
    const nextDelta = deltas[ed.m + 1][ed.c];
    const stat = layerStats[ed.m];
    const pct = (stat.relativeToOut * 100).toFixed(4);
    const isVanished = stat.relativeToOut < 0.005;

    return {
      badge: `가중치 ${stat.label} (${ed.conn})`,
      badgeClass: 'is-back',
      title: `${ed.key} · ${ed.fromLabel} → ${ed.toLabel}`,
      bodyHtml: `
        <p class="insp-lead">
          ${
            isVanished
              ? `<strong>기울기 소실 구간!</strong> 기울기가 <strong>${fmtSci(grad)}</strong>으로 너무 작아 1회 수정량(Δw)이 사실상 0에 가깝습니다.`
              : `앞 뉴런 신호(<strong>${fmt(prevAct, 4)}</strong>)와 뒤 뉴런 델타(<strong>${fmtSci(nextDelta)}</strong>)를 곱해 수정 방향과 크기를 정합니다.`
          }
        </p>
        <div class="insp-calc">
          <div class="insp-calc-row"><span>현재 가중치 (w)</span><strong class="num">${fmt(wOld, 6)}</strong></div>
          <div class="insp-calc-row"><span>앞 뉴런 신호 (${ed.fromLabel})</span><strong class="num text-fwd">${fmt(prevAct, 5)}</strong></div>
          <div class="insp-calc-row"><span>뒤 뉴런 델타 (δ)</span><strong class="num text-back">${fmtSci(nextDelta)}</strong></div>
          <div class="insp-calc-row"><span>기울기 (∂E/∂w)</span><strong class="num text-back">${fmtSci(grad)}</strong></div>
          <div class="insp-calc-row is-total">
            <span>수정 후 (lr=${lr.toFixed(2)})</span>
            <strong class="num">${fmt(wNew, 6)} (${deltaW >= 0 ? '+' : ''}${fmtSci(deltaW)})</strong>
          </div>
        </div>
        <p class="insp-tip">출력층(${layerStats[layerStats.length - 1].label}) 대비 이 층의 학습 신호 세기: <strong>${pct}%</strong></p>
      `,
    };
  }

  // Error boxes
  const isTotal = hit.id === 'E_total';
  const idx = hit.id === 'E_o2' ? 1 : 0;
  return {
    badge: '오차 계산 · 역전파 출발점',
    badgeClass: 'is-back',
    title: isTotal ? '전체 오차 (E_total)' : `출력 오차 (${hit.id})`,
    bodyHtml: isTotal
      ? `
        <p class="insp-lead">두 출력 오차를 합한 값입니다. <strong>${before.numHidden}개 은닉층(총 ${weights.length * 4}개 가중치)</strong> 전체가 이 숫자를 줄이려 동시에 움직입니다.</p>
        <div class="insp-calc">
          <div class="insp-calc-row"><span>E_o1</span><strong class="num">${fmt(before.errors[0], 6)}</strong></div>
          <div class="insp-calc-row"><span>E_o2</span><strong class="num">${fmt(before.errors[1], 6)}</strong></div>
          <div class="insp-calc-row is-total"><span>E_total = E_o1 + E_o2</span><strong class="num text-back">${fmt(before.eTotal, 6)}</strong></div>
          <div class="insp-calc-row"><span>1회 수정 후 예상 오차</span><strong class="num" style="color:var(--success)">${fmt(bp.after.eTotal, 6)}</strong></div>
        </div>
      `
      : `
        <p class="insp-lead">목표값과 출력값 차이를 제곱한 뒤 0.5를 곱해 오차를 잽니다.</p>
        <div class="insp-calc">
          <div class="insp-calc-row"><span>목표 t${idx + 1}</span><strong class="num">${fmt(before.targets[idx], 2)}</strong></div>
          <div class="insp-calc-row"><span>현재 출력 o${idx + 1}</span><strong class="num text-fwd">${fmt(before.out[idx], 6)}</strong></div>
          <div class="insp-calc-row is-total"><span>${hit.id} = 0.5 × (목표 − 출력)²</span><strong class="num text-back">${fmt(before.errors[idx], 6)}</strong></div>
        </div>
      `,
  };
}

export function mountDeep10Layers({ rootEl, onStatsChange }) {
  const state = {
    numHiddenLayers: DEEP_DEFAULT_CONFIG.numHiddenLayers,
    activation: DEEP_DEFAULT_CONFIG.activation,
    inputs: [...DEEP_DEFAULT_CONFIG.inputs],
    targets: [...DEEP_DEFAULT_CONFIG.targets],
    lr: DEEP_DEFAULT_CONFIG.lr,
    weights: createInitialWeights(DEEP_DEFAULT_CONFIG.numHiddenLayers),
    flowMode: 'backward',
    loupeOn: false,
    theaterOn: false,
    activeCameraPreset: 'fit',
    inspected: null,
    appliedUpdates: 0,
    errorHistory: [],
    isLiveTraining: false,
    liveTimer: null,
    liveTick: 0,
    visitedKeywords: new Set(['kw_deep10']),
  };

  rootEl.innerHTML = `
    <div class="ambient-mesh" aria-hidden="true">
      <div class="ambient-orb is-orb-1"></div>
      <div class="ambient-orb is-orb-2"></div>
      <div class="ambient-orb is-orb-3"></div>
    </div>

    <!-- 상단 컴팩트 히어로 + 4대 핵심 키워드 바 -->
    <section class="hero-compact" id="deep-hero">
      <canvas class="hero-canvas" id="deep-entropy-canvas" aria-hidden="true"></canvas>
      <div class="hero-compact-inner glass-panel">
        <div class="hero-title-group">
          <div class="hero-eyebrow-row">
            <span class="hero-pill-badge">심화 확장 실험실 · 은닉층 10개 (총 12층 · 가중치 44개)</span>
            <span class="kw-mastery-counter" id="deep-kw-counter">핵심 개념 1 / 4 탐험</span>
          </div>
          <h1 class="hero-main-title">
            은닉층을 <span class="text-fwd">#10개</span>로 깊게 쌓으면
            <span class="text-back">#기울기소실</span>은 왜 생기고,
            <span style="color:var(--success)">#ReLU</span>는 어떻게 살릴까?
          </h1>
        </div>

        <div class="keyword-mastery-bar deep-kw-bar" id="deep-kw-bar" role="navigation" aria-label="심화 핵심 키워드 바로가기"></div>
      </div>
    </section>

    <!-- 초대형 풀와이드(96vw) 10-은닉층 시뮬레이션 아레나 -->
    <section class="arena-section">
      <div class="arena-container" id="deep-arena-container">
        <div class="neural-studio glass-panel" id="deep-studio">
          <!-- 도구줄 1: 활성화 함수 · 흐름 모드 · 즉시 실행 -->
          <div class="studio-header">
            <div class="activation-switch" role="group" aria-label="은닉층 활성화 함수 선택">
              <span class="selector-label">은닉층 활성화</span>
              <button type="button" class="neu-pill is-active" id="deep-act-sigmoid" data-act="sigmoid">
                Sigmoid <small>기울기 소실</small>
              </button>
              <button type="button" class="neu-pill" id="deep-act-relu" data-act="relu">
                ReLU <small>신호 보존</small>
              </button>
            </div>

            <div class="mode-switch-group" role="group" aria-label="시뮬레이션 제어 및 학습 실행">
              <button type="button" class="mode-chip" id="deep-mode-fwd" data-mode="forward">순전파 →</button>
              <button type="button" class="mode-chip is-active-back" id="deep-mode-back" data-mode="backward">← 역전파</button>
              <button type="button" class="btn btn-primary btn-sm" id="deep-step-btn">⚡ 1회 역전파 발사</button>
              <button type="button" class="btn btn-back btn-sm" id="deep-live-btn">🚀 왕복 연속 학습</button>
              <button type="button" class="btn btn-ghost btn-sm" id="deep-reset-btn">초기화</button>
            </div>
          </div>

          <!-- 도구줄 2: 12층 카메라 포커스 점프 + 줌/돋보기/크게보기 -->
          <div class="camera-focus-bar" aria-label="12층 확대·축소 카메라 조작">
            <div class="camera-presets" role="group" aria-label="구간 빠른 이동">
              <span class="selector-label">카메라 이동</span>
              <button type="button" class="layer-jump-chip is-active" data-cam="fit">전체 12층 (0)</button>
              <button type="button" class="layer-jump-chip" data-cam="front">앞쪽 H1~H3</button>
              <button type="button" class="layer-jump-chip" data-cam="mid">중간 H4~H7</button>
              <button type="button" class="layer-jump-chip" data-cam="back">뒤쪽 H8~Out</button>
            </div>

            <div class="mode-switch-group" role="group" aria-label="돋보기 및 확대 축소">
              <button type="button" class="tool-chip" id="deep-loupe-btn" title="전체 보기에서도 커서 주변을 2.2배 확대해 내부 수치 투시 (단축키 Z)">🔍 돋보기 (Z)</button>
              <div class="zoom-group" role="group" aria-label="줌 배율">
                <button type="button" class="zoom-btn" id="deep-zoom-out" aria-label="축소">−</button>
                <button type="button" class="zoom-btn is-label" id="deep-zoom-reset" title="전체 맞춤으로 복원">100%</button>
                <button type="button" class="zoom-btn" id="deep-zoom-in" aria-label="확대">+</button>
              </div>
              <button type="button" class="tool-chip" id="deep-theater-btn" title="시뮬레이션 화면 최대 확대">⛶ 크게 보기</button>
            </div>
          </div>

          <!-- 1600×540 줌/팬/미니맵/돋보기 라이브 캔버스 -->
          <div class="canvas-stage" id="deep-canvas-stage">
            <canvas id="deep-neural-canvas" aria-label="은닉층 10개(총 12층) 딥러닝 순전파·역전파 줌/팬 시뮬레이션"></canvas>
            <div class="canvas-tooltip" id="deep-canvas-tooltip" hidden></div>

            <aside class="inspector-drawer glass-panel" id="deep-inspector-drawer" hidden aria-live="polite">
              <div class="insp-header">
                <div>
                  <span class="insp-badge" id="deep-insp-badge">-</span>
                  <h4 class="insp-title" id="deep-insp-title">-</h4>
                </div>
                <button type="button" class="btn btn-ghost btn-sm" id="deep-insp-close" aria-label="해설 닫기">✕</button>
              </div>
              <div class="insp-body" id="deep-insp-body"></div>
            </aside>
          </div>

          <!-- 층별 기울기 감쇠 이퀄라이저 차트 + 실시간 오차 요약 -->
          <div class="grad-dashboard">
            <div class="grad-chart-panel neu-card">
              <div class="grad-chart-head">
                <div>
                  <span class="metric-label">층별 역전파 기울기 크기 (막대를 클릭하면 해당 층으로 바로 확대됩니다)</span>
                  <div class="grad-status-callout" id="deep-grad-callout">-</div>
                </div>
                <div class="grad-ratio-pill num" id="deep-ratio-pill">W1/W11 = 0%</div>
              </div>
              <div class="grad-bars-grid" id="deep-grad-bars" role="group" aria-label="층별 기울기 감쇠 막대 차트"></div>
            </div>

            <div class="deep-live-metrics neu-card">
              <div>
                <span class="metric-label">전체 오차 추이 (목표: 0에 가깝게)</span>
                <div class="sparkline-head">
                  <strong class="num spark-val" id="deep-etotal-val">0.000000</strong>
                  <span class="delta-badge is-down num" id="deep-update-count">0회 수정</span>
                </div>
                <svg id="deep-sparkline-svg" viewBox="0 0 220 42" preserveAspectRatio="none" aria-label="오차 감소 추이 그래프"></svg>
              </div>

              <div class="deep-compare-mini">
                <div class="deep-mini-row">
                  <span>1회 역전파 후 예상 오차</span>
                  <strong class="num" id="deep-next-err" style="color:var(--success)">-</strong>
                </div>
                <div class="deep-mini-row">
                  <span>앞층 W1 평균 수정폭</span>
                  <strong class="num text-back" id="deep-w1-delta">-</strong>
                </div>
                <div class="deep-mini-row">
                  <span>출력층 끝 W 평균 수정폭</span>
                  <strong class="num text-fwd" id="deep-wout-delta">-</strong>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- 하단 2열 실험실: 실시간 슬라이더 + Sigmoid vs ReLU 핵심 비교 -->
        <div class="deep-bottom-grid">
          <div class="glass-panel deep-card-pad">
            <div class="controls-header">
              <div>
                <h4>실시간 구조 · 하이퍼파라미터 실험실</h4>
                <p class="deep-sub-note">은닉층 수를 <strong>1개</strong>로 내리면 기본 3층 시뮬레이션(M5)과 정확히 같은 값이 됩니다.</p>
              </div>
              <button type="button" class="btn btn-ghost btn-sm" id="deep-restore-btn">기본값 복원</button>
            </div>

            <div class="deep-preset-row" role="group" aria-label="추천 비교 실험">
              <button type="button" class="kw-tag is-back is-clickable" data-preset="sig10">10층 Sigmoid 소실</button>
              <button type="button" class="kw-tag is-clickable" style="color:var(--success);border-color:rgba(16,185,129,0.4)" data-preset="relu10">10층 ReLU 부활</button>
              <button type="button" class="kw-tag is-fwd is-clickable" data-preset="m5_1">1층 (기존 M5 비교)</button>
              <button type="button" class="kw-tag is-clickable" data-preset="fast_lr">학습률 0.50 가속</button>
            </div>

            <div class="sliders-grid">
              <label class="slider-item">
                <span class="slider-label-row">
                  <span>은닉층 개수 (깊이)</span>
                  <strong class="num" id="deep-val-depth">10층 (총 12층)</strong>
                </span>
                <input type="range" id="deep-sl-depth" min="1" max="10" step="1" value="10" />
              </label>

              <label class="slider-item">
                <span class="slider-label-row">
                  <span>학습률 (lr)</span>
                  <strong class="num" id="deep-val-lr">0.05</strong>
                </span>
                <input type="range" id="deep-sl-lr" min="0.01" max="1.0" step="0.01" value="0.05" />
              </label>

              <label class="slider-item">
                <span class="slider-label-row">
                  <span>입력 i1</span>
                  <strong class="num" id="deep-val-i1">0.05</strong>
                </span>
                <input type="range" id="deep-sl-i1" min="0" max="1" step="0.01" value="0.05" />
              </label>

              <label class="slider-item">
                <span class="slider-label-row">
                  <span>입력 i2</span>
                  <strong class="num" id="deep-val-i2">0.10</strong>
                </span>
                <input type="range" id="deep-sl-i2" min="0" max="1" step="0.01" value="0.10" />
              </label>
            </div>
          </div>

          <div class="glass-panel deep-card-pad">
            <div class="controls-header">
              <div>
                <h4>한눈에 비교: 왜 10층에서는 활성화 함수가 중요할까?</h4>
                <p class="deep-sub-note">역전파는 출력층에서 앞층으로 거슬러 올라가며 매 층의 기울기를 <strong>연달아 곱합니다</strong>.</p>
              </div>
            </div>

            <div class="deep-insight-cards">
              <div class="neu-card deep-insight-box is-warn">
                <div class="deep-insight-head">
                  <span class="kw-tag is-back">#Sigmoid · 기울기 소실</span>
                  <strong class="num text-back">최대 0.25</strong>
                </div>
                <p>시그모이드의 미분값은 아무리 커도 <strong>0.25 이하</strong>입니다. 은닉층 10개를 거슬러 올라가면 0.25가 10번 가까이 곱해져(약 <strong>0.000001배</strong>) 앞쪽 가중치 <strong>W1</strong>은 거의 움직이지 못합니다.</p>
              </div>

              <div class="neu-card deep-insight-box is-good">
                <div class="deep-insight-head">
                  <span class="kw-tag" style="color:var(--success);border-color:rgba(16,185,129,0.4)">#ReLU · 신호 보존</span>
                  <strong class="num" style="color:var(--success)">양수 구간 1.00</strong>
                </div>
                <p>ReLU는 양수 신호가 통과할 때 기울기가 그대로 <strong>1</strong>입니다. 10개 층을 거슬러 올라가도 신호가 사라지지 않아 <strong>W1까지 수십만 배 강하게</strong> 학습 신호가 도착합니다.</p>
              </div>
            </div>

            <details class="formula-details">
              <summary>연쇄 법칙(Chain Rule) 10층 수식 펼쳐 보기 (선택)</summary>
              <div class="formula-body">
                <p class="formula-line num">
                  ∂E / ∂W₁ = δ_out · [ ∏(l=10→1) W_(l+1) · σ'(net_l) ] · i
                </p>
                <p class="formula-line">
                  중간의 <strong>σ'(net_l)</strong>이 Sigmoid에서는 항상 ≤ 0.25이므로 곱셈이 쌓일수록 지수적으로 0에 수렴하고,
                  ReLU에서는 양수 구간에서 1이 되어 깊은 신경망(Deep Learning) 학습이 가능해집니다.
                </p>
              </div>
            </details>
          </div>
        </div>
      </div>
    </section>
  `;

  // 히어로 배경 파티클
  const heroCanvas = rootEl.querySelector('#deep-entropy-canvas');
  if (heroCanvas) mountEntropyCanvas(heroCanvas);

  // DOM 참조
  const arenaEl = rootEl.querySelector('#deep-arena-container');
  const canvasEl = rootEl.querySelector('#deep-neural-canvas');
  const tooltipEl = rootEl.querySelector('#deep-canvas-tooltip');
  const inspDrawerEl = rootEl.querySelector('#deep-inspector-drawer');
  const inspBadgeEl = rootEl.querySelector('#deep-insp-badge');
  const inspTitleEl = rootEl.querySelector('#deep-insp-title');
  const inspBodyEl = rootEl.querySelector('#deep-insp-body');
  const inspCloseBtn = rootEl.querySelector('#deep-insp-close');

  const kwBarEl = rootEl.querySelector('#deep-kw-bar');
  const kwCounterEl = rootEl.querySelector('#deep-kw-counter');
  const gradBarsEl = rootEl.querySelector('#deep-grad-bars');
  const gradCalloutEl = rootEl.querySelector('#deep-grad-callout');
  const ratioPillEl = rootEl.querySelector('#deep-ratio-pill');
  const eTotalValEl = rootEl.querySelector('#deep-etotal-val');
  const updateCountEl = rootEl.querySelector('#deep-update-count');
  const sparkSvgEl = rootEl.querySelector('#deep-sparkline-svg');
  const nextErrEl = rootEl.querySelector('#deep-next-err');
  const w1DeltaEl = rootEl.querySelector('#deep-w1-delta');
  const wOutDeltaEl = rootEl.querySelector('#deep-wout-delta');

  const zoomResetBtn = rootEl.querySelector('#deep-zoom-reset');
  const loupeBtn = rootEl.querySelector('#deep-loupe-btn');
  const theaterBtn = rootEl.querySelector('#deep-theater-btn');
  const liveBtn = rootEl.querySelector('#deep-live-btn');

  const slDepth = rootEl.querySelector('#deep-sl-depth');
  const slLr = rootEl.querySelector('#deep-sl-lr');
  const slI1 = rootEl.querySelector('#deep-sl-i1');
  const slI2 = rootEl.querySelector('#deep-sl-i2');
  const valDepth = rootEl.querySelector('#deep-val-depth');
  const valLr = rootEl.querySelector('#deep-val-lr');
  const valI1 = rootEl.querySelector('#deep-val-i1');
  const valI2 = rootEl.querySelector('#deep-val-i2');

  let latestBp = null;

  const neuralCanvas = createDeepNeuralCanvas(canvasEl, tooltipEl, {
    onInspectElement(hit) {
      state.inspected = hit;
      renderInspector();
    },
    onCameraChange({ relativePct }) {
      if (zoomResetBtn) zoomResetBtn.textContent = `${relativePct}%`;
    },
  });

  function computeBp() {
    latestBp = deepBackprop({
      inputs: state.inputs,
      targets: state.targets,
      weights: state.weights,
      activation: state.activation,
      lr: state.lr,
    });
    if (state.errorHistory.length === 0) {
      state.errorHistory = [latestBp.before.eTotal];
    }
    return latestBp;
  }

  function renderKeywordBar() {
    kwCounterEl.textContent = `핵심 개념 ${state.visitedKeywords.size} / ${DEEP_KEYWORDS.length} 탐험`;
    kwBarEl.innerHTML = DEEP_KEYWORDS.map((kw) => {
      const done = state.visitedKeywords.has(kw.id);
      return `
        <button type="button" class="kw-pill-card ${done ? 'is-mastered' : ''}" data-kw-action="${kw.action}" data-kw-id="${kw.id}">
          <span class="kw-pill-top">
            <strong>${kw.label}</strong>
            <span class="kw-check">${done ? '✓' : '→'}</span>
          </span>
          <span class="kw-pill-sub">${kw.sub}</span>
        </button>
      `;
    }).join('');
  }

  function renderInspector() {
    if (!state.inspected || !latestBp) {
      inspDrawerEl.hidden = true;
      return;
    }
    const content = buildDeepInspectorContent(state.inspected, latestBp, state.weights);
    inspBadgeEl.textContent = content.badge;
    inspBadgeEl.className = `insp-badge ${content.badgeClass}`;
    inspTitleEl.textContent = content.title;
    inspBodyEl.innerHTML = content.bodyHtml;
    inspDrawerEl.hidden = false;
  }

  function renderSparkline() {
    const pts = state.errorHistory;
    if (!pts.length) return;
    const minV = Math.min(...pts) * 0.92;
    const maxV = Math.max(...pts, minV + 0.01);
    const span = Math.max(1e-6, maxV - minV);
    const coords = pts.map((v, i) => {
      const x = pts.length === 1 ? 110 : 8 + (i / (pts.length - 1)) * 204;
      const y = 36 - ((v - minV) / span) * 28;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    const last = coords[coords.length - 1].split(',');
    sparkSvgEl.innerHTML = `
      <polyline fill="none" stroke="var(--success)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" points="${coords.join(' ')}" />
      <circle cx="${last[0]}" cy="${last[1]}" r="3.6" fill="var(--success)" />
    `;
  }

  function renderGradBars(bp) {
    const stats = bp.layerStats;
    const first = stats[0];
    const last = stats[stats.length - 1];
    const ratioPct = first.relativeToOut * 100;

    ratioPillEl.textContent = `${first.label}/${last.label} = ${
      ratioPct < 0.01 ? `${ratioPct.toFixed(5)}%` : `${ratioPct.toFixed(2)}%`
    }`;

    if (state.numHiddenLayers === 1) {
      gradCalloutEl.innerHTML = `<strong class="text-fwd">1-은닉층 모드(기존 M5 기준)</strong>: 층이 얕아 앞층 <strong>W1</strong>까지 신호가 바로 전달됩니다.`;
    } else if (first.relativeToOut < 0.005) {
      gradCalloutEl.innerHTML = `<strong class="text-back">기울기 소실 발생!</strong> ${state.numHiddenLayers}개 Sigmoid 층을 거치며 <strong>W1</strong> 기울기가 출력층 대비 <strong>${ratioPct.toFixed(5)}%</strong>로 꺼졌습니다.`;
    } else {
      gradCalloutEl.innerHTML = `<strong style="color:var(--success)">신호 생존!</strong> ReLU 덕분에 <strong>W1</strong>까지 출력층 대비 <strong>${ratioPct.toFixed(2)}%</strong> 세기의 학습 신호가 도착했습니다.`;
    }

    gradBarsEl.style.gridTemplateColumns = `repeat(${stats.length}, minmax(0, 1fr))`;
    gradBarsEl.innerHTML = stats
      .map((s) => {
        // 로그 스케일로 극적인 차이(1e-10 vs 1e-4 vs 1e-2)를 막대 높이에 반영
        const r = Math.max(1e-11, s.relativeToOut);
        const hPct =
          r >= 1
            ? 100
            : Math.max(5, Math.min(100, ((Math.log10(r) + 11) / 11) * 100));
        const isVanish = s.relativeToOut < 0.005;
        const barClass = isVanish
          ? 'is-vanish'
          : state.activation === 'relu'
            ? 'is-relu'
            : 'is-strong';
        const pctLabel =
          s.relativeToOut >= 0.1
            ? `${(s.relativeToOut * 100).toFixed(0)}%`
            : s.relativeToOut >= 0.001
              ? `${(s.relativeToOut * 100).toFixed(2)}%`
              : `${(s.relativeToOut * 100).toExponential(0)}`;

        return `
          <button type="button" class="grad-bar-col" data-focus-matrix="${s.matrixIdx}" title="${s.label} (${s.connection}) 평균 기울기 ${fmtSci(s.meanAbsGrad)} — 클릭해 확대">
            <span class="grad-bar-pct num">${pctLabel}</span>
            <span class="grad-bar-track">
              <span class="grad-bar-fill ${barClass}" style="height:${hPct.toFixed(1)}%"></span>
            </span>
            <strong class="grad-bar-lbl num">${s.label}</strong>
            <span class="grad-bar-conn">${s.connection}</span>
          </button>
        `;
      })
      .join('');
  }

  function syncUi({ resetFit = false } = {}) {
    const bp = computeBp();

    // 활성화 / 흐름 모드 버튼
    rootEl.querySelectorAll('[data-act]').forEach((btn) => {
      btn.classList.toggle('is-active', btn.dataset.act === state.activation);
    });
    rootEl.querySelector('#deep-mode-fwd').classList.toggle('is-active-fwd', state.flowMode === 'forward');
    rootEl.querySelector('#deep-mode-back').classList.toggle('is-active-back', state.flowMode === 'backward');

    // 슬라이더 라벨 동기화
    slDepth.value = String(state.numHiddenLayers);
    slLr.value = String(state.lr);
    slI1.value = String(state.inputs[0]);
    slI2.value = String(state.inputs[1]);
    valDepth.textContent = `${state.numHiddenLayers}층 (총 ${state.numHiddenLayers + 2}층)`;
    valLr.textContent = state.lr.toFixed(2);
    valI1.textContent = state.inputs[0].toFixed(2);
    valI2.textContent = state.inputs[1].toFixed(2);

    // 수치 메트릭
    const firstStat = bp.layerStats[0];
    const lastStat = bp.layerStats[bp.layerStats.length - 1];
    eTotalValEl.textContent = fmt(bp.before.eTotal, 6);
    updateCountEl.textContent = `${state.appliedUpdates}회 수정`;
    nextErrEl.textContent = `${fmt(bp.after.eTotal, 6)} (−${fmtSci(Math.max(0, bp.errorReduction))})`;
    w1DeltaEl.textContent = `Δ ≈ ${fmtSci(firstStat.meanAbsDeltaW)}`;
    wOutDeltaEl.textContent = `Δ ≈ ${fmtSci(lastStat.meanAbsDeltaW)}`;

    renderGradBars(bp);
    renderSparkline();
    renderInspector();

    neuralCanvas.setState({
      bp,
      weights: state.weights,
      mode: state.flowMode,
      activeInspect: state.inspected,
      resetFit,
    });

    if (onStatsChange) {
      onStatsChange({
        numHiddenLayers: state.numHiddenLayers,
        activation: state.activation,
        appliedUpdates: state.appliedUpdates,
        eTotal: bp.before.eTotal,
      });
    }
  }

  function applyOneStep(showNotice = true) {
    const bp = latestBp || computeBp();
    const reduction = bp.errorReduction;
    state.weights = cloneWeights(bp.nextWeights);
    state.appliedUpdates += 1;
    state.flowMode = 'backward';
    state.errorHistory.push(bp.after.eTotal);
    if (state.errorHistory.length > 80) state.errorHistory.shift();

    syncUi();
    neuralCanvas.triggerBackpropWave(reduction);
    flashElement(eTotalValEl);

    if (showNotice) {
      showToast(
        `${state.appliedUpdates}회 역전파 완료! (${state.numHiddenLayers}개 은닉층 · ${state.activation.toUpperCase()})`,
        'success'
      );
    }
  }

  function stopLiveTraining() {
    if (state.liveTimer) {
      clearInterval(state.liveTimer);
      state.liveTimer = null;
    }
    state.isLiveTraining = false;
    liveBtn.classList.remove('is-running');
    liveBtn.textContent = '🚀 왕복 연속 학습';
  }

  function toggleLiveTraining() {
    if (state.isLiveTraining) {
      stopLiveTraining();
      showToast('연속 학습을 멈췄습니다.', 'info');
      return;
    }
    state.isLiveTraining = true;
    state.liveTick = 0;
    liveBtn.classList.add('is-running');
    liveBtn.textContent = '⏹ 학습 멈추기';
    showToast('순전파(파랑) ↔ 역전파(주황) 왕복 연속 학습을 시작합니다!', 'success');

    state.liveTimer = setInterval(() => {
      state.liveTick += 1;
      if (state.liveTick % 2 === 1) {
        state.flowMode = 'forward';
        syncUi();
      } else {
        applyOneStep(false);
      }
    }, 680);
  }

  function setCameraPreset(preset) {
    state.activeCameraPreset = preset;
    rootEl.querySelectorAll('[data-cam]').forEach((b) => {
      b.classList.toggle('is-active', b.dataset.cam === String(preset));
    });
    neuralCanvas.focusRegion(preset, false);
  }

  function resetWeights(keepStructure = true) {
    stopLiveTraining();
    if (!keepStructure) {
      state.numHiddenLayers = DEEP_DEFAULT_CONFIG.numHiddenLayers;
      state.activation = DEEP_DEFAULT_CONFIG.activation;
      state.inputs = [...DEEP_DEFAULT_CONFIG.inputs];
      state.lr = DEEP_DEFAULT_CONFIG.lr;
    }
    state.weights = createInitialWeights(state.numHiddenLayers);
    state.appliedUpdates = 0;
    state.errorHistory = [];
    state.inspected = null;
    syncUi({ resetFit: !keepStructure });
  }

  // ===== 이벤트 바인딩 =====
  kwBarEl.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-kw-action]');
    if (!btn) return;
    const action = btn.dataset.kwAction;
    state.visitedKeywords.add(btn.dataset.kwId);
    renderKeywordBar();

    if (action === 'overview') {
      state.numHiddenLayers = 10;
      state.weights = createInitialWeights(10);
      state.appliedUpdates = 0;
      state.errorHistory = [];
      state.flowMode = 'forward';
      syncUi({ resetFit: true });
      setCameraPreset('fit');
      showToast('10개 은닉층(총 12층 · 44개 가중치) 전체 구조를 한눈에 봅니다.', 'info');
    } else if (action === 'vanish') {
      stopLiveTraining();
      state.numHiddenLayers = 10;
      state.activation = 'sigmoid';
      state.weights = createInitialWeights(10);
      state.appliedUpdates = 0;
      state.errorHistory = [];
      state.flowMode = 'backward';
      syncUi({ resetFit: true });
      setCameraPreset('front');
      showToast('Sigmoid 10층: 앞쪽 H1~H3의 기울기가 점선처럼 거의 사라졌습니다!', 'info');
    } else if (action === 'relu') {
      stopLiveTraining();
      state.numHiddenLayers = 10;
      state.activation = 'relu';
      state.weights = createInitialWeights(10);
      state.appliedUpdates = 0;
      state.errorHistory = [];
      state.flowMode = 'backward';
      syncUi({ resetFit: true });
      setCameraPreset('front');
      neuralCanvas.triggerBackpropWave(latestBp.errorReduction);
      showToast('ReLU 10층: 앞층 H1~H3까지 주황색 역전파 신호가 54만 배 강하게 살아났습니다!', 'success');
    } else if (action === 'compare') {
      stopLiveTraining();
      state.numHiddenLayers = state.numHiddenLayers === 1 ? 10 : 1;
      state.weights = createInitialWeights(state.numHiddenLayers);
      state.appliedUpdates = 0;
      state.errorHistory = [];
      state.flowMode = 'backward';
      syncUi({ resetFit: true });
      setCameraPreset('fit');
      showToast(
        state.numHiddenLayers === 1
          ? '1-은닉층(기본 M5와 동일)으로 전환했습니다. 다시 누르면 10층으로 돌아옵니다.'
          : '10-은닉층(총 12층)으로 확장했습니다!',
        'info'
      );
    }
  });

  // 활성화 함수 토글
  rootEl.querySelectorAll('[data-act]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const nextAct = btn.dataset.act;
      if (state.activation === nextAct) return;
      stopLiveTraining();
      state.activation = nextAct;
      state.weights = createInitialWeights(state.numHiddenLayers);
      state.appliedUpdates = 0;
      state.errorHistory = [];
      state.flowMode = 'backward';
      syncUi();
      neuralCanvas.triggerBackpropWave(latestBp.errorReduction);
      showToast(
        nextAct === 'relu'
          ? 'ReLU 활성화 적용: 앞층 W1까지 기울기가 살아남는 모습을 비교해 보세요!'
          : 'Sigmoid 활성화 적용: 깊은 층을 지날수록 앞층 기울기가 0에 가까워집니다.',
        'success'
      );
    });
  });

  // 순전파 / 역전파 모드
  rootEl.querySelectorAll('[data-mode]').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.flowMode = btn.dataset.mode;
      syncUi();
    });
  });

  // 실행 버튼
  rootEl.querySelector('#deep-step-btn').addEventListener('click', () => applyOneStep(true));
  liveBtn.addEventListener('click', toggleLiveTraining);
  rootEl.querySelector('#deep-reset-btn').addEventListener('click', () => {
    resetWeights(true);
    showToast('현재 층 구조의 가중치를 초기값으로 되돌렸습니다.', 'info');
  });

  // 카메라 포커스 점프
  rootEl.querySelectorAll('[data-cam]').forEach((btn) => {
    btn.addEventListener('click', () => setCameraPreset(btn.dataset.cam));
  });

  // 줌 · 돋보기 · 크게보기
  rootEl.querySelector('#deep-zoom-in').addEventListener('click', () => neuralCanvas.zoomByFactor(1.24));
  rootEl.querySelector('#deep-zoom-out').addEventListener('click', () => neuralCanvas.zoomByFactor(1 / 1.24));
  zoomResetBtn.addEventListener('click', () => setCameraPreset('fit'));

  function setLoupe(next) {
    state.loupeOn = next;
    loupeBtn.classList.toggle('is-active', next);
    loupeBtn.textContent = next ? '🔍 돋보기 ON' : '🔍 돋보기 (Z)';
    neuralCanvas.setLoupe(next);
  }
  loupeBtn.addEventListener('click', () => {
    setLoupe(!state.loupeOn);
    showToast(
      state.loupeOn
        ? 'X-Ray 돋보기 ON: 캔버스 위를 움직이면 전체 보기에서도 각 뉴런의 net·δ·기울기를 투시합니다!'
        : '돋보기를 껐습니다.',
      'info'
    );
  });

  theaterBtn.addEventListener('click', () => {
    state.theaterOn = !state.theaterOn;
    arenaEl.classList.toggle('is-theater', state.theaterOn);
    theaterBtn.classList.toggle('is-active', state.theaterOn);
    theaterBtn.textContent = state.theaterOn ? '🗗 기본 크기' : '⛶ 크게 보기';
  });

  inspCloseBtn.addEventListener('click', () => {
    state.inspected = null;
    inspDrawerEl.hidden = true;
    neuralCanvas.setState({ bp: latestBp, weights: state.weights, mode: state.flowMode, activeInspect: null });
  });

  // 기울기 막대 클릭 -> 해당 층으로 부드럽게 확대 이동 + 첫 가중치 해부
  gradBarsEl.addEventListener('click', (e) => {
    const col = e.target.closest('[data-focus-matrix]');
    if (!col) return;
    const mIdx = Number(col.dataset.focusMatrix);
    rootEl.querySelectorAll('[data-cam]').forEach((b) => b.classList.remove('is-active'));
    neuralCanvas.focusRegion(mIdx + 1, false);
    state.inspected = {
      kind: 'edge',
      id: `edge_${mIdx}_0_0`,
      ref: {
        id: `edge_${mIdx}_0_0`,
        m: mIdx,
        r: 0,
        c: 0,
        fromLabel: mIdx === 0 ? 'i1' : `h${mIdx},1`,
        toLabel: mIdx === state.numHiddenLayers ? 'o1' : `h${mIdx + 1},1`,
        key: `W${mIdx + 1}[1,1]`,
        conn: latestBp.layerStats[mIdx].connection,
      },
    };
    syncUi();
  });

  // 슬라이더 & 프리셋
  slDepth.addEventListener('input', () => {
    stopLiveTraining();
    state.numHiddenLayers = Number(slDepth.value);
    state.weights = createInitialWeights(state.numHiddenLayers);
    state.appliedUpdates = 0;
    state.errorHistory = [];
    state.inspected = null;
    syncUi({ resetFit: true });
    setCameraPreset('fit');
  });
  slLr.addEventListener('input', () => {
    state.lr = Number(slLr.value);
    syncUi();
  });
  slI1.addEventListener('input', () => {
    state.inputs[0] = Number(slI1.value);
    syncUi();
  });
  slI2.addEventListener('input', () => {
    state.inputs[1] = Number(slI2.value);
    syncUi();
  });

  rootEl.querySelector('#deep-restore-btn').addEventListener('click', () => {
    resetWeights(false);
    setCameraPreset('fit');
    showToast('10-은닉층 Sigmoid 기본 실험 상태로 복원했습니다.', 'info');
  });

  rootEl.querySelectorAll('[data-preset]').forEach((btn) => {
    btn.addEventListener('click', () => {
      stopLiveTraining();
      const key = btn.dataset.preset;
      if (key === 'sig10') {
        state.numHiddenLayers = 10;
        state.activation = 'sigmoid';
        state.lr = 0.05;
      } else if (key === 'relu10') {
        state.numHiddenLayers = 10;
        state.activation = 'relu';
        state.lr = 0.05;
      } else if (key === 'm5_1') {
        state.numHiddenLayers = 1;
        state.activation = 'sigmoid';
        state.lr = 0.05;
      } else if (key === 'fast_lr') {
        state.numHiddenLayers = 10;
        state.activation = 'relu';
        state.lr = 0.5;
      }
      state.inputs = [...DEEP_DEFAULT_CONFIG.inputs];
      state.weights = createInitialWeights(state.numHiddenLayers);
      state.appliedUpdates = 0;
      state.errorHistory = [];
      state.flowMode = 'backward';
      state.inspected = null;
      syncUi({ resetFit: true });
      setCameraPreset('fit');
      neuralCanvas.triggerBackpropWave(latestBp.errorReduction);
    });
  });

  // 키보드 단축키 (Z: 돋보기, 0: 전체 맞춤, +/-: 줌, Esc: 인스펙터 닫기)
  window.addEventListener('keydown', (e) => {
    if (e.target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
    if (e.key === 'z' || e.key === 'Z') {
      setLoupe(!state.loupeOn);
    } else if (e.key === '0') {
      setCameraPreset('fit');
    } else if (e.key === '+' || e.key === '=') {
      neuralCanvas.zoomByFactor(1.2);
    } else if (e.key === '-') {
      neuralCanvas.zoomByFactor(1 / 1.2);
    } else if (e.key === 'Escape' && state.inspected) {
      state.inspected = null;
      inspDrawerEl.hidden = true;
      neuralCanvas.setState({ bp: latestBp, weights: state.weights, mode: state.flowMode, activeInspect: null });
    }
  });

  renderKeywordBar();
  syncUi({ resetFit: true });

  return {
    destroy() {
      stopLiveTraining();
      neuralCanvas.destroy();
    },
  };
}
