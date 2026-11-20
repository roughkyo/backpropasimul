/**
 * 하단 챕터 카드용 실시간 인터랙티브 도식(SVG) 렌더러
 * 모든 색상을 CSS 변수로 참조하여 화이트 모드 / 다크 모드 모두에서 선명하게 보입니다.
 */
import { fmt, sigmoid } from '../math/backprop.js';

export function renderChapter1Diagram(before) {
  const rows = [
    { label: '은닉 h1', val: before.h[0] },
    { label: '은닉 h2', val: before.h[1] },
    { label: '출력 o1', val: before.o[0] },
    { label: '출력 o2', val: before.o[1] },
  ];
  return `
    <div class="mini-diagram">
      <div class="mini-diagram-title">현재 순전파 신호 크기 (0 ~ 1)</div>
      <div class="bar-diagram">
        ${rows
          .map(
            (r) => `
          <div class="bar-diag-row">
            <span>${r.label}</span>
            <div class="bar-diag-track">
              <div class="bar-diag-fill is-fwd" style="width:${(r.val * 100).toFixed(1)}%"></div>
            </div>
            <strong class="num">${fmt(r.val, 4)}</strong>
          </div>
        `
          )
          .join('')}
      </div>
    </div>
  `;
}

export function renderChapter2Diagram(before) {
  const renderGap = (name, out, target, err) => {
    const oPct = (out * 100).toFixed(1);
    const tPct = (target * 100).toFixed(1);
    const left = Math.min(out, target) * 100;
    const width = Math.abs(out - target) * 100;
    return `
      <div class="gap-row">
        <div class="gap-head">
          <span>${name} (출력 ${fmt(out, 3)} / 목표 ${fmt(target, 2)})</span>
          <strong class="num text-back">E = ${fmt(err, 4)}</strong>
        </div>
        <div class="gap-track">
          <div class="gap-span" style="left:${left.toFixed(1)}%;width:${Math.max(1, width).toFixed(1)}%"></div>
          <div class="gap-dot is-target" style="left:${tPct}%" title="목표"></div>
          <div class="gap-dot is-output" style="left:${oPct}%" title="출력"></div>
        </div>
      </div>
    `;
  };
  return `
    <div class="mini-diagram">
      <div class="mini-diagram-title">
        <span>목표와 출력 사이의 오차 간극</span>
        <strong class="num text-back">E_total = ${fmt(before.eTotal, 6)}</strong>
      </div>
      ${renderGap('o1', before.o[0], before.targets[0], before.errors[0])}
      ${renderGap('o2', before.o[1], before.targets[1], before.errors[1])}
    </div>
  `;
}

export function renderChapter3Diagram(selected) {
  const oPos = 24 + selected.outVal * 292;
  const tPos = 24 + selected.targetVal * 292;
  const needDecrease = selected.step1 > 0.0005;
  const needIncrease = selected.step1 < -0.0005;
  const dirLabel = needDecrease
    ? '출력이 목표보다 큼 → 출력을 줄여야 함 (+)'
    : needIncrease
      ? '출력이 목표보다 작음 → 출력을 키워야 함 (−)'
      : '목표와 거의 일치';

  return `
    <div class="mini-diagram">
      <div class="mini-diagram-title">
        <span>${selected.meta.to} 오차 방향 도식</span>
        <strong class="num text-back">${selected.step1 >= 0 ? '+' : ''}${fmt(selected.step1, 6)}</strong>
      </div>
      <svg viewBox="0 0 340 76" class="diag-svg" role="img" aria-label="목표와 출력 위치 비교">
        <defs>
          <marker id="diag-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--back)"></path>
          </marker>
        </defs>
        <line x1="24" y1="42" x2="316" y2="42" stroke="var(--border-strong)" stroke-width="2" />
        <text x="24" y="64" fill="var(--text-dim)" font-size="10" text-anchor="middle">0.0</text>
        <text x="170" y="64" fill="var(--text-dim)" font-size="10" text-anchor="middle">0.5</text>
        <text x="316" y="64" fill="var(--text-dim)" font-size="10" text-anchor="middle">1.0</text>

        <line x1="${oPos.toFixed(1)}" y1="42" x2="${tPos.toFixed(1)}" y2="42" stroke="var(--back)" stroke-width="3.2" marker-end="url(#diag-arrow)" />

        <circle cx="${tPos.toFixed(1)}" cy="42" r="5.5" fill="var(--text)" />
        <text x="${tPos.toFixed(1)}" y="24" fill="var(--text)" font-size="10.5" font-weight="700" text-anchor="middle">목표 ${fmt(selected.targetVal, 2)}</text>

        <circle cx="${oPos.toFixed(1)}" cy="42" r="6" fill="var(--fwd)" />
        <text x="${oPos.toFixed(1)}" y="24" fill="var(--fwd)" font-size="10.5" font-weight="700" text-anchor="middle">${selected.meta.to} ${fmt(selected.outVal, 2)}</text>
      </svg>
      <div class="mini-diagram-foot">${dirLabel}</div>
    </div>
  `;
}

export function renderChapter4Diagram(before, selected) {
  const netVal = before.netO[selected.meta.toIdx];
  const outVal = selected.outVal;
  const slope = selected.step2;

  const w = 340;
  const h = 112;
  const padL = 28;
  const padR = 16;
  const padT = 14;
  const padB = 22;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const xToPx = (x) => padL + ((x + 6) / 12) * plotW;
  const yToPx = (y) => padT + (1 - y) * plotH;

  const pts = [];
  for (let i = 0; i <= 48; i++) {
    const x = -6 + (i / 48) * 12;
    pts.push(`${i === 0 ? 'M' : 'L'} ${xToPx(x).toFixed(1)} ${yToPx(sigmoid(x)).toFixed(1)}`);
  }

  const cx = Math.max(-5.8, Math.min(5.8, netVal));
  const cy = outVal;
  const dx = 1.8;
  const tx1 = cx - dx;
  const ty1 = Math.max(0, Math.min(1, cy - slope * dx));
  const tx2 = cx + dx;
  const ty2 = Math.max(0, Math.min(1, cy + slope * dx));

  return `
    <div class="mini-diagram">
      <div class="mini-diagram-title">
        <span>시그모이드 곡선과 현재 접선 기울기</span>
        <strong class="num text-back">기울기 = ${fmt(slope, 6)}</strong>
      </div>
      <svg viewBox="0 0 ${w} ${h}" class="diag-svg" role="img" aria-label="시그모이드 접선 기울기 그래프">
        <line x1="${padL}" y1="${yToPx(0.5)}" x2="${w - padR}" y2="${yToPx(0.5)}" stroke="var(--border-strong)" stroke-dasharray="3 3" />
        <line x1="${xToPx(0)}" y1="${padT}" x2="${xToPx(0)}" y2="${h - padB}" stroke="var(--border-strong)" stroke-dasharray="3 3" />
        <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}" stroke="var(--border-strong)" />

        <text x="6" y="${yToPx(1) + 4}" fill="var(--text-dim)" font-size="9.5">1.0</text>
        <text x="6" y="${yToPx(0.5) + 4}" fill="var(--text-dim)" font-size="9.5">0.5</text>
        <text x="6" y="${yToPx(0) + 4}" fill="var(--text-dim)" font-size="9.5">0.0</text>
        <text x="${xToPx(0)}" y="${h - 6}" fill="var(--text-dim)" font-size="9.5" text-anchor="middle">net = 0 (최대 기울기 0.25)</text>

        <path d="${pts.join(' ')}" fill="none" stroke="var(--fwd)" stroke-width="2.4" />
        <line x1="${xToPx(tx1).toFixed(1)}" y1="${yToPx(ty1).toFixed(1)}" x2="${xToPx(tx2).toFixed(1)}" y2="${yToPx(ty2).toFixed(1)}" stroke="var(--back)" stroke-width="2.8" stroke-linecap="round" />
        <circle cx="${xToPx(cx).toFixed(1)}" cy="${yToPx(cy).toFixed(1)}" r="5.5" fill="var(--back)" stroke="var(--surface-solid)" stroke-width="1.6" />
      </svg>
      <div class="mini-diagram-foot">
        현재 ${selected.meta.to} = ${fmt(outVal, 4)} → 최대 민감도(0.25)의 ${((slope / 0.25) * 100).toFixed(0)}% 수준
      </div>
    </div>
  `;
}

export function renderChapter5Diagram(selected) {
  const sigPct = (selected.step3 * 100).toFixed(1);
  return `
    <div class="mini-diagram">
      <div class="mini-diagram-title">
        <span>${selected.meta.label} 신호 합류</span>
        <strong class="num text-fwd">${selected.meta.from} = ${fmt(selected.step3, 6)}</strong>
      </div>
      <div class="confluence-box">
        <div class="conf-node is-fwd">
          <small>앞쪽 은닉 노드</small>
          <strong>${selected.meta.from} = ${fmt(selected.step3, 4)}</strong>
        </div>
        <div class="conf-arrow is-fwd">→</div>
        <div class="conf-center">
          <small>수정 대상</small>
          <strong>${selected.meta.key}</strong>
        </div>
        <div class="conf-arrow is-back">←</div>
        <div class="conf-node is-back">
          <small>뒤쪽 오차×민감도</small>
          <strong>${fmt(selected.step1 * selected.step2, 4)}</strong>
        </div>
      </div>
      <div class="bar-diag-row" style="margin-top:8px">
        <span>신호 세기</span>
        <div class="bar-diag-track">
          <div class="bar-diag-fill is-fwd" style="width:${sigPct}%"></div>
        </div>
        <strong class="num">${sigPct}%</strong>
      </div>
    </div>
  `;
}
