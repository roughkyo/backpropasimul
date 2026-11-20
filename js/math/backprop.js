/**
 * M5 역전파 한 걸음씩 — 순수 계산 모듈
 * 노트북 3장(셀 53~71)의 2-2-2 시그모이드 신경망(편향 없음) 기준.
 *
 * 연결 정의 (노트북 셀 53, 68, 69 코드 기준):
 *   W1 = [[w1, w2], [w3, w4]]  =>  w1: i1->h1, w2: i1->h2, w3: i2->h1, w4: i2->h2
 *   W2 = [[w5, w6], [w7, w8]]  =>  w5: h1->o1, w6: h1->o2, w7: h2->o1, w8: h2->o2
 */

export const DEFAULT_PARAMS = Object.freeze({
  inputs: [0.05, 0.1],
  targets: [0.01, 0.99],
  W1: [
    [0.15, 0.2],
    [0.25, 0.3],
  ],
  W2: [
    [0.35, 0.4],
    [0.45, 0.5],
  ],
  lr: 0.05,
});

export const WEIGHT_META = Object.freeze({
  w5: { key: 'w5', row: 0, col: 0, from: 'h1', to: 'o1', fromIdx: 0, toIdx: 0, label: 'w5 (h1 → o1)' },
  w6: { key: 'w6', row: 0, col: 1, from: 'h1', to: 'o2', fromIdx: 0, toIdx: 1, label: 'w6 (h1 → o2)' },
  w7: { key: 'w7', row: 1, col: 0, from: 'h2', to: 'o1', fromIdx: 1, toIdx: 0, label: 'w7 (h2 → o1)' },
  w8: { key: 'w8', row: 1, col: 1, from: 'h2', to: 'o2', fromIdx: 1, toIdx: 1, label: 'w8 (h2 → o2)' },
});

export function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

export function sigmoidDerivativeFromOutput(out) {
  return out * (1 - out);
}

export function cloneMatrix(m) {
  return m.map((row) => [...row]);
}

/**
 * 순전파 + 오차 계산
 */
export function forwardPass({ inputs, targets, W1, W2 }) {
  const [i1, i2] = inputs;
  const [t1, t2] = targets;

  const net_h1 = i1 * W1[0][0] + i2 * W1[1][0];
  const net_h2 = i1 * W1[0][1] + i2 * W1[1][1];
  const h1 = sigmoid(net_h1);
  const h2 = sigmoid(net_h2);

  const net_o1 = h1 * W2[0][0] + h2 * W2[1][0];
  const net_o2 = h1 * W2[0][1] + h2 * W2[1][1];
  const o1 = sigmoid(net_o1);
  const o2 = sigmoid(net_o2);

  const e1 = 0.5 * (t1 - o1) ** 2;
  const e2 = 0.5 * (t2 - o2) ** 2;
  const eTotal = e1 + e2;

  return {
    inputs: [i1, i2],
    targets: [t1, t2],
    netH: [net_h1, net_h2],
    h: [h1, h2],
    netO: [net_o1, net_o2],
    o: [o1, o2],
    errors: [e1, e2],
    eTotal,
  };
}

/**
 * 출력층 가중치 하나(w5~w8)에 대한 역전파 3단계 계산
 */
export function computeWeightBackprop(forward, W2, weightKey, lr) {
  const meta = WEIGHT_META[weightKey];
  if (!meta) {
    throw new Error(`Unknown weight key: ${weightKey}`);
  }

  const outVal = forward.o[meta.toIdx];
  const targetVal = forward.targets[meta.toIdx];
  const hiddenVal = forward.h[meta.fromIdx];
  const oldWeight = W2[meta.row][meta.col];

  // [1단계] ∂E / ∂out = -(target - out)
  const step1 = -(targetVal - outVal);
  // [2단계] ∂out / ∂net = out * (1 - out)
  const step2 = sigmoidDerivativeFromOutput(outVal);
  // [3단계] ∂net / ∂w = h_from
  const step3 = hiddenVal;

  const gradient = step1 * step2 * step3;
  const deltaWeight = -lr * gradient;
  const newWeight = oldWeight + deltaWeight;

  return {
    meta,
    oldWeight,
    newWeight,
    deltaWeight,
    step1,
    step2,
    step3,
    gradient,
    lr,
    targetVal,
    outVal,
    hiddenVal,
  };
}

/**
 * 선택한 단일 가중치(w5~w8) 또는 출력층 전체(w5~w8)를 업데이트한 전후 비교
 */
export function simulateUpdate(params, targetWeight = 'w5', updateAllOutputWeights = false) {
  const before = forwardPass(params);
  const details = {};
  for (const key of ['w5', 'w6', 'w7', 'w8']) {
    details[key] = computeWeightBackprop(before, params.W2, key, params.lr);
  }

  const nextW2 = cloneMatrix(params.W2);
  const keysToApply = updateAllOutputWeights ? ['w5', 'w6', 'w7', 'w8'] : [targetWeight];
  for (const key of keysToApply) {
    const d = details[key];
    nextW2[d.meta.row][d.meta.col] = d.newWeight;
  }

  const after = forwardPass({
    ...params,
    W2: nextW2,
  });

  return {
    before,
    after,
    selected: details[targetWeight],
    allDetails: details,
    appliedKeys: keysToApply,
    nextW2,
    errorReduction: before.eTotal - after.eTotal,
  };
}

/**
 * 은닉층(W1) 포함 전체 1스텝 역전파 (다회 반복 학습 데모 및 개념퀴즈2 확인용)
 */
export function stepFullBackprop(params, includeHiddenLayer = false) {
  const fwd = forwardPass(params);
  const nextW2 = cloneMatrix(params.W2);
  const nextW1 = cloneMatrix(params.W1);

  const dOut = [
    -(fwd.targets[0] - fwd.o[0]) * sigmoidDerivativeFromOutput(fwd.o[0]),
    -(fwd.targets[1] - fwd.o[1]) * sigmoidDerivativeFromOutput(fwd.o[1]),
  ];

  // W2 업데이트
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 2; c++) {
      const grad = dOut[c] * fwd.h[r];
      nextW2[r][c] = params.W2[r][c] - params.lr * grad;
    }
  }

  if (includeHiddenLayer) {
    for (let r = 0; r < 2; r++) {
      for (let hIdx = 0; hIdx < 2; hIdx++) {
        const dE_dh = dOut[0] * params.W2[hIdx][0] + dOut[1] * params.W2[hIdx][1];
        const dE_dnetH = dE_dh * sigmoidDerivativeFromOutput(fwd.h[hIdx]);
        const grad = dE_dnetH * fwd.inputs[r];
        nextW1[r][hIdx] = params.W1[r][hIdx] - params.lr * grad;
      }
    }
  }

  const nextParams = { ...params, W1: nextW1, W2: nextW2 };
  const after = forwardPass(nextParams);
  return { nextParams, before: fwd, after };
}

export function fmt(num, digits = 6) {
  if (!Number.isFinite(num)) return 'NaN';
  return num.toFixed(digits);
}
