/**
 * 시뮬레이션 요소 클릭 시 열리는 상세 부연설명 인스펙터
 * 신경망의 모든 요소(입력 i1~i2, 은닉 h1~h2, 출력 o1~o2, 가중치 w1~w8, 오차 E_o1/E_o2/E_total)를
 * 클릭하면 현재 숫자가 대입된 실제 계산 과정과 직관적 부연설명을 보여 줍니다.
 */
import { fmt, sigmoid, sigmoidDerivativeFromOutput } from '../math/backprop.js';

function miniSigmoidSvg(netVal, outVal, slope) {
  const w = 260;
  const h = 86;
  const padL = 22;
  const padR = 12;
  const padT = 10;
  const padB = 18;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const xToPx = (x) => padL + ((x + 6) / 12) * plotW;
  const yToPx = (y) => padT + (1 - y) * plotH;

  const pts = [];
  for (let i = 0; i <= 40; i++) {
    const x = -6 + (i / 40) * 12;
    pts.push(`${i === 0 ? 'M' : 'L'} ${xToPx(x).toFixed(1)} ${yToPx(sigmoid(x)).toFixed(1)}`);
  }
  const cx = Math.max(-5.8, Math.min(5.8, netVal));
  const dx = 1.8;
  const tx1 = cx - dx;
  const ty1 = Math.max(0, Math.min(1, outVal - slope * dx));
  const tx2 = cx + dx;
  const ty2 = Math.max(0, Math.min(1, outVal + slope * dx));

  return `
    <svg viewBox="0 0 ${w} ${h}" class="insp-svg" role="img" aria-label="시그모이드 위치 및 기울기">
      <line x1="${padL}" y1="${yToPx(0.5)}" x2="${w - padR}" y2="${yToPx(0.5)}" stroke="var(--border-strong)" stroke-dasharray="3 3" />
      <line x1="${xToPx(0)}" y1="${padT}" x2="${xToPx(0)}" y2="${h - padB}" stroke="var(--border-strong)" stroke-dasharray="3 3" />
      <path d="${pts.join(' ')}" fill="none" stroke="var(--fwd)" stroke-width="2.2" />
      <line x1="${xToPx(tx1).toFixed(1)}" y1="${yToPx(ty1).toFixed(1)}" x2="${xToPx(tx2).toFixed(1)}" y2="${yToPx(ty2).toFixed(1)}" stroke="var(--back)" stroke-width="2.6" stroke-linecap="round" />
      <circle cx="${xToPx(cx).toFixed(1)}" cy="${yToPx(outVal).toFixed(1)}" r="4.8" fill="var(--back)" stroke="var(--surface-solid)" stroke-width="1.5" />
      <text x="${padL}" y="${h - 4}" fill="var(--text-dim)" font-size="9">net=${fmt(netVal, 3)}</text>
      <text x="${w - padR}" y="${h - 4}" fill="var(--back)" font-size="9" text-anchor="end">기울기 ${fmt(slope, 4)}</text>
    </svg>
  `;
}

export function buildInspectorHtml(targetId, { sim, W1, W2, lr }) {
  const { before, allDetails } = sim;

  // 1) 입력 노드 i1, i2
  if (targetId === 'i1' || targetId === 'i2') {
    const idx = targetId === 'i1' ? 0 : 1;
    const val = before.inputs[idx];
    const wA = idx === 0 ? { k: 'w1', v: W1[0][0], to: 'h1' } : { k: 'w3', v: W1[1][0], to: 'h1' };
    const wB = idx === 0 ? { k: 'w2', v: W1[0][1], to: 'h2' } : { k: 'w4', v: W1[1][1], to: 'h2' };
    return {
      badge: '입력층 노드',
      tone: 'fwd',
      title: `입력 ${targetId} = ${fmt(val, 2)}`,
      body: `
        <p class="insp-lead">신경망에 가장 먼저 들어오는 데이터 값입니다. 이 값은 두 가중치 선을 타고 은닉층(h1, h2)으로 나뉘어 전달됩니다.</p>
        <div class="insp-calc">
          <div class="insp-calc-row"><span>${wA.to}로 전달되는 신호</span><strong class="num">${fmt(val, 2)} × ${fmt(wA.v, 2)} (${wA.k}) = ${fmt(val * wA.v, 5)}</strong></div>
          <div class="insp-calc-row"><span>${wB.to}로 전달되는 신호</span><strong class="num">${fmt(val, 2)} × ${fmt(wB.v, 2)} (${wB.k}) = ${fmt(val * wB.v, 5)}</strong></div>
        </div>
        <p class="insp-tip">아래 실험실의 입력 슬라이더로 ${targetId} 값을 바꾸면, 선을 따라 흐르는 파란 신호와 최종 출력이 즉시 변합니다.</p>
      `,
    };
  }

  // 2) 은닉 노드 h1, h2
  if (targetId === 'h1' || targetId === 'h2') {
    const idx = targetId === 'h1' ? 0 : 1;
    const net = before.netH[idx];
    const out = before.h[idx];
    const slope = sigmoidDerivativeFromOutput(out);
    const w1Key = idx === 0 ? 'w1' : 'w2';
    const w2Key = idx === 0 ? 'w3' : 'w4';
    const w1Val = W1[0][idx];
    const w2Val = W1[1][idx];
    const outWeights = idx === 0 ? 'w5(→o1), w6(→o2)' : 'w7(→o1), w8(→o2)';
    return {
      badge: '은닉층 노드 · 시그모이드',
      tone: 'fwd',
      title: `은닉 ${targetId} = ${fmt(out, 6)}`,
      body: `
        <p class="insp-lead">두 입력 신호를 가중치와 곱해 모두 더한 뒤(가중합), 시그모이드 함수로 0~1 사이 값으로 바꾼 결과입니다.</p>
        <div class="insp-calc">
          <div class="insp-calc-row"><span>① 가중합 (net_${targetId})</span><strong class="num">${fmt(before.inputs[0], 2)}×${fmt(w1Val, 2)}(${w1Key}) + ${fmt(before.inputs[1], 2)}×${fmt(w2Val, 2)}(${w2Key}) = ${fmt(net, 6)}</strong></div>
          <div class="insp-calc-row"><span>② 시그모이드 통과</span><strong class="num">σ(${fmt(net, 4)}) = ${fmt(out, 6)}</strong></div>
          <div class="insp-calc-row"><span>③ 역전파 3단계 역할</span><strong>${outWeights}의 들어온 신호값으로 사용됨</strong></div>
        </div>
        ${miniSigmoidSvg(net, out, slope)}
      `,
    };
  }

  // 3) 출력 노드 o1, o2
  if (targetId === 'o1' || targetId === 'o2') {
    const idx = targetId === 'o1' ? 0 : 1;
    const net = before.netO[idx];
    const out = before.o[idx];
    const target = before.targets[idx];
    const err = before.errors[idx];
    const step1 = -(target - out);
    const slope = sigmoidDerivativeFromOutput(out);
    const wA = idx === 0 ? { k: 'w5', v: W2[0][0] } : { k: 'w6', v: W2[0][1] };
    const wB = idx === 0 ? { k: 'w7', v: W2[1][0] } : { k: 'w8', v: W2[1][1] };
    const direction = step1 > 0 ? '목표보다 커서 출력을 낮춰야 합니다 (+)' : '목표보다 작아서 출력을 높여야 합니다 (−)';

    return {
      badge: '출력층 노드 · 시그모이드',
      tone: 'back',
      title: `출력 ${targetId} = ${fmt(out, 6)}`,
      body: `
        <p class="insp-lead">신경망이 내놓은 최종 예측값입니다. 목표값(${fmt(target, 2)})과 비교해 오차 신호가 여기서부터 거꾸로 출발합니다.</p>
        <div class="insp-calc">
          <div class="insp-calc-row"><span>① 가중합 (net_${targetId})</span><strong class="num">${fmt(before.h[0], 4)}×${fmt(wA.v, 4)}(${wA.k}) + ${fmt(before.h[1], 4)}×${fmt(wB.v, 4)}(${wB.k}) = ${fmt(net, 6)}</strong></div>
          <div class="insp-calc-row"><span>② 예측값 / 목표값</span><strong class="num">${fmt(out, 6)} / 목표 ${fmt(target, 2)} (오차 ${fmt(err, 4)})</strong></div>
          <div class="insp-calc-row"><span>③ 역전파 1단계(오차 방향)</span><strong class="num text-back">${step1 >= 0 ? '+' : ''}${fmt(step1, 6)}</strong></div>
          <div class="insp-calc-row"><span>④ 역전파 2단계(민감도)</span><strong class="num text-back">${fmt(slope, 6)} (최대 0.25의 ${((slope / 0.25) * 100).toFixed(0)}%)</strong></div>
        </div>
        ${miniSigmoidSvg(net, out, slope)}
        <p class="insp-tip">${direction}</p>
      `,
    };
  }

  // 4) 출력층 가중치 w5 ~ w8
  if (allDetails[targetId]) {
    const d = allDetails[targetId];
    const dirWord = d.deltaWeight < 0 ? '줄어듭니다 (↓)' : '커집니다 (↑)';
    const whyDir =
      d.step1 > 0
        ? `출력 ${d.meta.to}이(가) 목표(${fmt(d.targetVal, 2)})보다 크므로, 연결 세기를 줄여 출력을 낮춥니다.`
        : `출력 ${d.meta.to}이(가) 목표(${fmt(d.targetVal, 2)})보다 작으므로, 연결 세기를 키워 출력을 높입니다.`;
    return {
      badge: '출력층 가중치 · 역전파 핵심',
      tone: 'back',
      title: `${d.meta.label} = ${fmt(d.oldWeight, 6)}`,
      actionWeight: d.meta.key,
      body: `
        <p class="insp-lead">${whyDir}</p>
        <div class="insp-calc">
          <div class="insp-calc-row"><span>[1단계] 얼마나 틀렸나</span><strong class="num text-back">${d.step1 >= 0 ? '+' : ''}${fmt(d.step1, 6)}</strong></div>
          <div class="insp-calc-row"><span>[2단계] ${d.meta.to} 민감도</span><strong class="num text-back">${fmt(d.step2, 6)}</strong></div>
          <div class="insp-calc-row"><span>[3단계] 들어온 신호 (${d.meta.from})</span><strong class="num text-fwd">${fmt(d.step3, 6)}</strong></div>
          <div class="insp-calc-row is-total"><span>세 값의 곱 (기울기)</span><strong class="num">${fmt(d.gradient, 6)}</strong></div>
          <div class="insp-calc-row"><span>학습률 ${lr.toFixed(2)} 적용 시</span><strong class="num">${fmt(d.oldWeight, 6)} → ${fmt(d.newWeight, 6)} (${dirWord})</strong></div>
        </div>
        <div class="insp-actions">
          <button type="button" class="btn btn-primary btn-sm" data-insp-apply="${d.meta.key}">${d.meta.key} 지금 1회 수정해 보기</button>
        </div>
      `,
    };
  }

  // 5) 은닉층 가중치 w1 ~ w4
  const w1Map = {
    w1: { r: 0, c: 0, from: 'i1', to: 'h1', inIdx: 0, hIdx: 0 },
    w2: { r: 0, c: 1, from: 'i1', to: 'h2', inIdx: 0, hIdx: 1 },
    w3: { r: 1, c: 0, from: 'i2', to: 'h1', inIdx: 1, hIdx: 0 },
    w4: { r: 1, c: 1, from: 'i2', to: 'h2', inIdx: 1, hIdx: 1 },
  };
  if (w1Map[targetId]) {
    const m = w1Map[targetId];
    const oldW = W1[m.r][m.c];
    const inVal = before.inputs[m.inIdx];
    const hVal = before.h[m.hIdx];
    // 앞쪽 층 역전파 실제 계산 (o1, o2 양쪽 오차 합산)
    const dOut0 = -(before.targets[0] - before.o[0]) * sigmoidDerivativeFromOutput(before.o[0]);
    const dOut1 = -(before.targets[1] - before.o[1]) * sigmoidDerivativeFromOutput(before.o[1]);
    const dE_dh = dOut0 * W2[m.hIdx][0] + dOut1 * W2[m.hIdx][1];
    const dE_dnetH = dE_dh * sigmoidDerivativeFromOutput(hVal);
    const grad = dE_dnetH * inVal;
    const newW = oldW - lr * grad;

    return {
      badge: '은닉층 가중치 (앞쪽 층 · 심화)',
      tone: 'fwd',
      title: `${targetId} (${m.from} → ${m.to}) = ${fmt(oldW, 4)}`,
      body: `
        <p class="insp-lead">앞쪽 층 가중치는 o1과 o2 두 출력에서 돌아오는 오차를 모두 합쳐서 전달받습니다. 이때 시그모이드 민감도(0.25 이하)가 한 번 더 곱해져 기울기가 훨씬 작아집니다.</p>
        <div class="insp-calc">
          <div class="insp-calc-row"><span>전달되는 입력 신호</span><strong class="num">${m.from}(${fmt(inVal, 2)}) × ${fmt(oldW, 2)} = ${fmt(inVal * oldW, 5)}</strong></div>
          <div class="insp-calc-row"><span>${m.to}에 모인 역전파 오차</span><strong class="num text-back">${fmt(dE_dh, 6)}</strong></div>
          <div class="insp-calc-row"><span>${targetId} 기울기 (∂E/∂${targetId})</span><strong class="num">${fmt(grad, 6)}</strong></div>
          <div class="insp-calc-row"><span>업데이트 시 변화</span><strong class="num">${fmt(oldW, 6)} → ${fmt(newW, 6)}</strong></div>
        </div>
        <p class="insp-tip">노트북 개념퀴즈2(셀 73)와 연결됩니다: w5(${fmt(allDetails.w5.gradient, 5)})보다 기울기가 훨씬 작다는 점을 비교해 보세요.</p>
      `,
    };
  }

  // 6) 오차 박스 E_o1, E_o2, E_total
  if (targetId === 'E_o1' || targetId === 'E_o2' || targetId === 'E_total') {
    if (targetId === 'E_total') {
      return {
        badge: '전체 오차 (손실 함수)',
        tone: 'back',
        title: `전체 오차 E_total = ${fmt(before.eTotal, 6)}`,
        body: `
          <p class="insp-lead">두 출력 노드에서 생긴 오차를 모두 더한 값입니다. 신경망 학습의 목표는 이 숫자를 0에 가깝게 줄이는 것입니다.</p>
          <div class="insp-calc">
            <div class="insp-calc-row"><span>o1 오차 (E_o1)</span><strong class="num">0.5 × (${fmt(before.targets[0], 2)} − ${fmt(before.o[0], 4)})² = ${fmt(before.errors[0], 6)}</strong></div>
            <div class="insp-calc-row"><span>o2 오차 (E_o2)</span><strong class="num">0.5 × (${fmt(before.targets[1], 2)} − ${fmt(before.o[1], 4)})² = ${fmt(before.errors[1], 6)}</strong></div>
            <div class="insp-calc-row is-total"><span>합계 (E_total)</span><strong class="num text-back">${fmt(before.eTotal, 6)}</strong></div>
          </div>
          <p class="insp-tip">앞에 0.5를 곱해 두는 이유: 제곱을 미분할 때 내려오는 숫자 2와 약분되어 역전파 1단계 식이 깔끔하게 -(목표 - 출력)으로 정리되기 때문입니다.</p>
        `,
      };
    }
    const idx = targetId === 'E_o1' ? 0 : 1;
    const oKey = idx === 0 ? 'o1' : 'o2';
    const diff = before.targets[idx] - before.o[idx];
    return {
      badge: '개별 출력 오차',
      tone: 'back',
      title: `${targetId} = ${fmt(before.errors[idx], 6)}`,
      body: `
        <p class="insp-lead">출력 ${oKey}가 목표값(${fmt(before.targets[idx], 2)})에서 얼마나 벗어났는지 제곱해서 계산한 오차입니다.</p>
        <div class="insp-calc">
          <div class="insp-calc-row"><span>차이 (목표 − 출력)</span><strong class="num">${fmt(before.targets[idx], 2)} − ${fmt(before.o[idx], 4)} = ${fmt(diff, 6)}</strong></div>
          <div class="insp-calc-row"><span>제곱 후 0.5 곱하기</span><strong class="num text-back">0.5 × (${fmt(diff, 4)})² = ${fmt(before.errors[idx], 6)}</strong></div>
        </div>
        <p class="insp-tip">차이를 제곱하므로 크게 틀릴수록 오차가 훨씬 커지고, 부호(+/−)가 사라져 항상 0 이상의 값이 됩니다.</p>
      `,
    };
  }

  return null;
}
