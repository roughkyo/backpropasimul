/**
 * 10-은닉층 딥러닝 역전파 엔진 자동 수치 검증 스크립트
 * 1) 1-은닉층 설정이 기존 M5 노트북 기준값과 소수점 6자리까지 일치하는지 검증
 * 2) 10-은닉층(총 44개 가중치) 전체에 대해 해석적 역전파 기울기 vs 중앙 유한차분(ε=1e-6) 대조
 *    - Sigmoid & ReLU(Leaky) 양쪽 모두 오차 < 1e-7 검증
 * 3) Sigmoid 10층에서의 기울기 소실(W1 << W11) vs ReLU 10층에서의 기울기 보존 확인
 */
import assert from 'node:assert/strict';
import {
  DEEP_DEFAULT_CONFIG,
  cloneWeights,
  createInitialWeights,
  deepBackprop,
  deepForward,
} from '../js/math/deep-backprop.js';

// 1) 1-은닉층 === 기존 M5 기준값 대조
const w1Layer = createInitialWeights(1);
const res1 = deepBackprop({
  inputs: DEEP_DEFAULT_CONFIG.inputs,
  targets: DEEP_DEFAULT_CONFIG.targets,
  weights: w1Layer,
  activation: 'sigmoid',
  lr: 0.05,
});
assert.ok(Math.abs(Number(res1.before.eTotal.toFixed(6)) - 0.245532) <= 1e-6, 'M5 E_total mismatch');
assert.ok(Math.abs(Number(res1.grads[1][0][0].toFixed(6)) - 0.071978) <= 1e-6, 'M5 ∂E/∂w5 mismatch');
assert.ok(Math.abs(Number(res1.nextWeights[1][0][0].toFixed(6)) - 0.346401) <= 1e-6, 'M5 w5 new mismatch');
console.log('[PASS] 1-은닉층 모드 === 기존 M5 노트북 기준값 100% 일치');

// 2) 10-은닉층 (총 44개 가중치) 유한차분 검증 (Sigmoid & ReLU)
for (const activation of ['sigmoid', 'relu']) {
  const weights = createInitialWeights(10);
  const bp = deepBackprop({
    inputs: DEEP_DEFAULT_CONFIG.inputs,
    targets: DEEP_DEFAULT_CONFIG.targets,
    weights,
    activation,
    lr: 0.05,
  });

  const eps = 1e-6;
  let maxErr = 0;
  let checkedCount = 0;

  for (let m = 0; m < weights.length; m++) {
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 2; c++) {
        const wPlus = cloneWeights(weights);
        const wMinus = cloneWeights(weights);
        wPlus[m][r][c] += eps;
        wMinus[m][r][c] -= eps;

        const ePlus = deepForward({
          inputs: DEEP_DEFAULT_CONFIG.inputs,
          targets: DEEP_DEFAULT_CONFIG.targets,
          weights: wPlus,
          activation,
        }).eTotal;
        const eMinus = deepForward({
          inputs: DEEP_DEFAULT_CONFIG.inputs,
          targets: DEEP_DEFAULT_CONFIG.targets,
          weights: wMinus,
          activation,
        }).eTotal;

        const numericGrad = (ePlus - eMinus) / (2 * eps);
        const analyticGrad = bp.grads[m][r][c];
        const diff = Math.abs(numericGrad - analyticGrad);
        if (diff > maxErr) maxErr = diff;

        assert.ok(
          diff < 1e-7,
          `[${activation}] W${m + 1}[${r}][${c}] grad mismatch: analytic=${analyticGrad}, numeric=${numericGrad}, diff=${diff}`
        );
        checkedCount++;
      }
    }
  }

  const ratioFirstToLast = bp.layerStats[0].relativeToOut;
  console.log(
    `[PASS] 10-은닉층 (${activation.toUpperCase()}): ${checkedCount}개 가중치 유한차분 일치 (maxErr=${maxErr.toExponential(
      2
    )}) | W1/W11 기울기 비율 = ${(ratioFirstToLast * 100).toFixed(6)}%`
  );
}

// 3) Sigmoid vs ReLU 기울기 소실 대비 검증
const sigBp = deepBackprop({
  inputs: DEEP_DEFAULT_CONFIG.inputs,
  targets: DEEP_DEFAULT_CONFIG.targets,
  weights: createInitialWeights(10),
  activation: 'sigmoid',
  lr: 0.05,
});
const reluBp = deepBackprop({
  inputs: DEEP_DEFAULT_CONFIG.inputs,
  targets: DEEP_DEFAULT_CONFIG.targets,
  weights: createInitialWeights(10),
  activation: 'relu',
  lr: 0.05,
});
assert.ok(
  reluBp.layerStats[0].meanAbsGrad > sigBp.layerStats[0].meanAbsGrad * 100,
  'ReLU should dramatically alleviate vanishing gradient at W1 compared to Sigmoid'
);
console.log(
  `[PASS] 기울기 소실 대비 검증: W1 평균 기울기 Sigmoid=${sigBp.layerStats[0].meanAbsGrad.toExponential(
    2
  )} vs ReLU=${reluBp.layerStats[0].meanAbsGrad.toExponential(2)}`
);
console.log('\n✅ 10-은닉층 딥러닝 역전파 수학 엔진 검증 완료!');
