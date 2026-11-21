/**
 * 10-은닉층(총 12층) 딥러닝 순전파·역전파 순수 계산 엔진
 * - 입력층(2) -> 은닉층 H1..Hk(각 2노드, 기본 k=10, 범위 1..10) -> 출력층(2)
 * - 활성화 함수: 'sigmoid' (기울기 소실 관찰) / 'relu' (기울기 보존 비교)
 * - 모든 층의 가중합(net), 출력(act), 역전파 델타(delta), 가중치 기울기(grad),
 *   층별 기울기 크기(meanAbsGrad, relativePct)를 계산합니다.
 */

export const DEEP_DEFAULT_CONFIG = Object.freeze({
  numHiddenLayers: 10,
  activation: 'sigmoid', // 'sigmoid' | 'relu'
  inputs: [0.05, 0.1],
  targets: [0.01, 0.99],
  lr: 0.05,
});

const LEAKY_ALPHA = 0.05;

export function sigmoid(x) {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

export function activate(x, kind = 'sigmoid') {
  if (kind === 'relu') {
    return x > 0 ? x : LEAKY_ALPHA * x;
  }
  return sigmoid(x);
}

export function activateDerivative(net, out, kind = 'sigmoid') {
  if (kind === 'relu') {
    return net > 0 ? 1 : LEAKY_ALPHA;
  }
  return out * (1 - out);
}

/**
 * 결정적 난수 생성기 (Mulberry32) — 언제 열어도 동일한 초기 가중치 재현
 */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * numHiddenLayers(1..10)에 대한 결정적 초기 가중치 행렬 목록 생성
 * 총 행렬 개수 = numHiddenLayers + 1 (W_0: In->H1, ..., W_k: Hk->Out)
 * - numHiddenLayers === 1일 때는 기존 M5(셀 53)와 100% 동일한 W1, W2를 사용!
 * - 마지막 출력층 행렬은 항상 M5의 W2([[0.35, 0.40], [0.45, 0.50]])로 통일해 연결감을 유지.
 */
export function createInitialWeights(numHiddenLayers = 10, seed = 20261121) {
  const k = Math.max(1, Math.min(10, Math.round(numHiddenLayers)));
  if (k === 1) {
    return [
      [
        [0.15, 0.2],
        [0.25, 0.3],
      ],
      [
        [0.35, 0.4],
        [0.45, 0.5],
      ],
    ];
  }

  const rand = mulberry32(seed);
  const weights = [];

  // 첫 입력층 -> H1
  weights.push([
    [0.15, 0.2],
    [0.25, 0.3],
  ]);

  // 은닉층 간 연결 (H1->H2 .. H_{k-1}->H_k): 양수 중심의 안정적 값 (0.25 ~ 0.65)
  for (let l = 1; l < k; l++) {
    const w00 = Number((0.28 + rand() * 0.34).toFixed(2));
    const w01 = Number((0.22 + rand() * 0.34).toFixed(2));
    const w10 = Number((0.24 + rand() * 0.34).toFixed(2));
    const w11 = Number((0.30 + rand() * 0.34).toFixed(2));
    weights.push([
      [w00, w01],
      [w10, w11],
    ]);
  }

  // 마지막 은닉층 H_k -> 출력층 Out (M5와 동일한 [[0.35, 0.40], [0.45, 0.50]])
  weights.push([
    [0.35, 0.4],
    [0.45, 0.5],
  ]);

  return weights;
}

export function cloneWeights(weights) {
  return weights.map((mat) => mat.map((row) => [...row]));
}

/**
 * 전체 층 순전파
 * layers[0] = 입력층, layers[1..k] = 은닉층 H1..Hk, layers[k+1] = 출력층
 */
export function deepForward({ inputs, targets, weights, activation = 'sigmoid' }) {
  const numMatrices = weights.length;
  const numHidden = numMatrices - 1;
  const nets = [null]; // 인덱스 0(입력층)은 net 없음
  const acts = [[inputs[0], inputs[1]]];

  for (let m = 0; m < numMatrices; m++) {
    const prev = acts[m];
    const W = weights[m];
    const n0 = prev[0] * W[0][0] + prev[1] * W[1][0];
    const n1 = prev[0] * W[0][1] + prev[1] * W[1][1];
    nets.push([n0, n1]);

    // 출력층은 항상 시그모이드(0~1 확률/목표 0.01~0.99 비교), 은닉층은 선택한 activation 적용
    const layerActKind = m === numMatrices - 1 ? 'sigmoid' : activation;
    acts.push([activate(n0, layerActKind), activate(n1, layerActKind)]);
  }

  const out = acts[acts.length - 1];
  const e0 = 0.5 * (targets[0] - out[0]) ** 2;
  const e1 = 0.5 * (targets[1] - out[1]) ** 2;
  const eTotal = e0 + e1;

  return {
    numHidden,
    numLayers: acts.length, // numHidden + 2
    activation,
    inputs: [...inputs],
    targets: [...targets],
    nets,
    acts,
    out: [...out],
    errors: [e0, e1],
    eTotal,
  };
}

/**
 * 전체 층 역전파 (출력층 -> H_k -> ... -> H1) + 층별 기울기 감쇠 통계
 */
export function deepBackprop({ inputs, targets, weights, activation = 'sigmoid', lr = 0.05 }) {
  const fwd = deepForward({ inputs, targets, weights, activation });
  const numMatrices = weights.length; // L = numHidden + 1
  const deltas = new Array(numMatrices + 1).fill(null); // deltas[1..L]
  const grads = new Array(numMatrices).fill(null); // grads[0..L-1] (행렬 m의 2x2 기울기)
  const nextWeights = cloneWeights(weights);

  // 1) 출력층 델타 (m = numMatrices - 1 -> layer index = numMatrices)
  const outIdx = numMatrices;
  const outAct = fwd.acts[outIdx];
  const outNet = fwd.nets[outIdx];
  const dE_dout = [-(targets[0] - outAct[0]), -(targets[1] - outAct[1])];
  const outDeriv = [
    activateDerivative(outNet[0], outAct[0], 'sigmoid'),
    activateDerivative(outNet[1], outAct[1], 'sigmoid'),
  ];
  deltas[outIdx] = [dE_dout[0] * outDeriv[0], dE_dout[1] * outDeriv[1]];

  // 2) 출력층부터 입력층 방향으로 거슬러 올라가며 기울기 및 앞층 델타 계산
  for (let m = numMatrices - 1; m >= 0; m--) {
    const prevAct = fwd.acts[m];
    const nextDelta = deltas[m + 1];
    const W = weights[m];

    const G = [
      [prevAct[0] * nextDelta[0], prevAct[0] * nextDelta[1]],
      [prevAct[1] * nextDelta[0], prevAct[1] * nextDelta[1]],
    ];
    grads[m] = G;

    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 2; c++) {
        nextWeights[m][r][c] = W[r][c] - lr * G[r][c];
      }
    }

    // 앞쪽 은닉층(m >= 1) 델타 계산
    if (m >= 1) {
      const hAct = fwd.acts[m];
      const hNet = fwd.nets[m];
      const errToH0 = nextDelta[0] * W[0][0] + nextDelta[1] * W[0][1];
      const errToH1 = nextDelta[0] * W[1][0] + nextDelta[1] * W[1][1];
      const deriv0 = activateDerivative(hNet[0], hAct[0], activation);
      const deriv1 = activateDerivative(hNet[1], hAct[1], activation);
      deltas[m] = [errToH0 * deriv0, errToH1 * deriv1];
    }
  }

  // 3) 층별 기울기 통계 (기울기 소실 시각화용)
  const outMeanGrad =
    (Math.abs(grads[numMatrices - 1][0][0]) +
      Math.abs(grads[numMatrices - 1][0][1]) +
      Math.abs(grads[numMatrices - 1][1][0]) +
      Math.abs(grads[numMatrices - 1][1][1])) /
    4;

  const layerStats = grads.map((G, m) => {
    const absVals = [
      Math.abs(G[0][0]),
      Math.abs(G[0][1]),
      Math.abs(G[1][0]),
      Math.abs(G[1][1]),
    ];
    const meanAbsGrad = (absVals[0] + absVals[1] + absVals[2] + absVals[3]) / 4;
    const maxAbsGrad = Math.max(...absVals);
    const relativeToOut = outMeanGrad > 1e-18 ? meanAbsGrad / outMeanGrad : 0;
    const fromLabel = m === 0 ? 'In' : `H${m}`;
    const toLabel = m === numMatrices - 1 ? 'Out' : `H${m + 1}`;
    return {
      matrixIdx: m,
      label: `W${m + 1}`,
      connection: `${fromLabel}→${toLabel}`,
      meanAbsGrad,
      maxAbsGrad,
      meanAbsDeltaW: meanAbsGrad * lr,
      relativeToOut,
    };
  });

  const after = deepForward({
    inputs,
    targets,
    weights: nextWeights,
    activation,
  });

  return {
    before: fwd,
    after,
    deltas,
    grads,
    nextWeights,
    layerStats,
    dE_dout,
    outDeriv,
    lr,
    errorReduction: fwd.eTotal - after.eTotal,
  };
}

export function fmtSci(val) {
  if (!Number.isFinite(val)) return 'NaN';
  const a = Math.abs(val);
  if (a === 0) return '0.000000';
  if (a < 1e-4) return val.toExponential(2);
  return val.toFixed(6);
}

export function fmt(val, digits = 6) {
  if (!Number.isFinite(val)) return 'NaN';
  return val.toFixed(digits);
}
