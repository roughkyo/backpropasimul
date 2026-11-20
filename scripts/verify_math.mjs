/**
 * 설계명세서 7장 / 부록 A-5 기준값 자동 대조 스크립트
 * 실행: node Deeplearning/sim/scripts/verify_math.mjs
 */
import assert from 'node:assert/strict';
import { DEFAULT_PARAMS, simulateUpdate } from '../js/math/backprop.js';

function assertClose6(label, actual, expected) {
  const actualRounded = Number(actual.toFixed(6));
  const diff = Math.abs(actualRounded - expected);
  assert.ok(
    diff <= 1e-6,
    `[FAIL] ${label}: expected ${expected.toFixed(6)}, got ${actual.toFixed(6)} (diff=${diff})`
  );
  console.log(`[PASS] ${label}: ${actual.toFixed(6)} === ${expected.toFixed(6)}`);
}

// 1) w5 단일 업데이트 기준값 확인 (셀 56, 58, 60, 62, 64, 66, 67)
const simW5 = simulateUpdate(DEFAULT_PARAMS, 'w5', false);
assertClose6('연습 신경망 출력 o1 (셀 56)', simW5.before.o[0], 0.600451);
assertClose6('연습 신경망 출력 o2 (셀 56)', simW5.before.o[1], 0.612599);
assertClose6('초기 전체 오차 E_total (셀 58)', simW5.before.eTotal, 0.245532);
assertClose6('[1단계] ∂E/∂out_o1 (셀 60)', simW5.selected.step1, 0.590451);
assertClose6('[2단계] out_o1(1-out_o1) (셀 62)', simW5.selected.step2, 0.239910);
assertClose6('[3단계] h1 (셀 64)', simW5.selected.step3, 0.508124);
assertClose6('∂E/∂w5 (셀 66)', simW5.selected.gradient, 0.071978);
assertClose6('w5 업데이트 (lr=0.05, 셀 67)', simW5.selected.newWeight, 0.346401);

// 2) w5~w8 모두 업데이트 후 기준값 확인 (셀 69, 71)
const simAll = simulateUpdate(DEFAULT_PARAMS, 'w5', true);
assertClose6('w6 업데이트 (셀 69)', simAll.allDetails.w6.newWeight, 0.402276);
assertClose6('w7 업데이트 (셀 69)', simAll.allDetails.w7.newWeight, 0.446388);
assertClose6('w8 업데이트 (셀 69)', simAll.allDetails.w8.newWeight, 0.502284);
assertClose6('w5~w8 업데이트 후 E_total (셀 71)', simAll.after.eTotal, 0.244804);

console.log('\n✅ 모든 부록 A-5 기준값이 소수점 6자리까지 100% 일치합니다.');
