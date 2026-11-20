/**
 * 공통 UI 유틸리티 — 토스트, 제출 모달, 값 변화 강조
 */
import { isSupabaseConfigured } from './storage.js';

let toastTimer = null;

export function showToast(message, type = 'info') {
  const el = document.getElementById('toast');
  if (!el) return;
  el.textContent = message;
  el.className = `toast is-visible is-${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.classList.remove('is-visible');
  }, 3200);
}

export function flashElement(el) {
  if (!el) return;
  el.classList.remove('is-flash');
  // reflow 트리거
  void el.offsetWidth;
  el.classList.add('is-flash');
}

export function formatKoreaTime(isoStr) {
  if (!isoStr) return '-';
  try {
    return new Date(isoStr).toLocaleString('ko-KR', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return isoStr;
  }
}

export function initSubmitModal(store, onSubmitted) {
  const backdrop = document.getElementById('submit-modal-backdrop');
  const form = document.getElementById('submit-form');
  const idInput = document.getElementById('submit-student-id');
  const nameInput = document.getElementById('submit-student-name');
  const pwInput = document.getElementById('submit-password');
  const summaryEl = document.getElementById('submit-summary-box');
  const statusEl = document.getElementById('submit-status-msg');
  const submitBtn = document.getElementById('submit-confirm-btn');
  const closeBtns = document.querySelectorAll('[data-close-submit-modal]');
  const modeNote = document.getElementById('submit-mode-note');

  function openModal() {
    const mod = store.getModuleState('m5');
    const predDone = Boolean(mod.prediction?.choiceId);
    const missionCount = Object.values(mod.missions || {}).filter(Boolean).length;
    const checkCorrect = Object.values(mod.checks || {}).filter((c) => c?.correct).length;
    const hasReflection = Boolean(mod.reflection?.trim());

    idInput.value = store.state.studentProfile?.studentId || '';
    nameInput.value = store.state.studentProfile?.studentName || '';
    pwInput.value = '';
    statusEl.hidden = true;
    statusEl.textContent = '';

    summaryEl.innerHTML = `
      <div class="summary-chip ${predDone ? 'is-done' : ''}">예측: ${predDone ? '완료' : '미완료'}</div>
      <div class="summary-chip ${missionCount >= 2 ? 'is-done' : ''}">미션: ${missionCount}/2 달성</div>
      <div class="summary-chip ${checkCorrect >= 2 ? 'is-done' : ''}">확인 체크: ${checkCorrect}/2 정답</div>
      <div class="summary-chip ${hasReflection ? 'is-done' : ''}">생각 질문: ${hasReflection ? '작성함' : '선택'}</div>
    `;

    modeNote.textContent = isSupabaseConfigured()
      ? '입력한 비밀번호는 암호화(해시)되어 저장되며, 같은 학번으로 다시 제출할 때 본인 확인에만 사용됩니다.'
      : '현재 Supabase 연결 전(로컬 데모 모드)입니다. config.js에 프로젝트 정보를 입력하면 교사 DB로 자동 전송됩니다.';

    backdrop.hidden = false;
    setTimeout(() => (idInput.value ? pwInput.focus() : idInput.focus()), 50);
  }

  function closeModal() {
    backdrop.hidden = true;
  }

  closeBtns.forEach((btn) => btn.addEventListener('click', closeModal));
  backdrop.addEventListener('click', (e) => {
    if (e.target === backdrop) closeModal();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !backdrop.hidden) closeModal();
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    submitBtn.disabled = true;
    submitBtn.textContent = '제출 중...';
    statusEl.hidden = true;

    const result = await store.submit({
      studentId: idInput.value,
      studentName: nameInput.value,
      password: pwInput.value,
    });

    submitBtn.disabled = false;
    submitBtn.textContent = '작업 기록 제출';

    if (result.ok) {
      statusEl.hidden = false;
      statusEl.className = 'submit-status is-success';
      statusEl.textContent = `${result.message} (제출 횟수: ${result.submit_count}회)`;
      showToast('작업 기록이 제출되었습니다!', 'success');
      if (onSubmitted) onSubmitted(result);
    } else {
      statusEl.hidden = false;
      statusEl.className = 'submit-status is-error';
      statusEl.textContent = result.message || '제출 중 오류가 발생했습니다.';
    }
  });

  return { openModal, closeModal };
}
