/**
 * 앱 진입점 — 상단 스크롤 프로그레스 바, 플로팅 헤더, 발표 모드, 제출 모달, M5 마운트
 */
import { APP_CONFIG } from '../config.js';
import { LearningStore } from './core/storage.js';
import { formatKoreaTime, initSubmitModal, showToast } from './core/ui.js';
import { mountM5Module } from './modules/m5-backprop.js';

const store = new LearningStore();

const titleEl = document.getElementById('app-title');
const progressBadgeEl = document.getElementById('header-progress-badge');
const submitStatusEl = document.getElementById('header-submit-status');
const openSubmitBtn = document.getElementById('header-submit-btn');
const presentBtn = document.getElementById('toggle-present-btn');
const scrollBarEl = document.getElementById('scroll-progress-bar');
const themeBtns = document.querySelectorAll('[data-set-theme]');

function applyTheme(theme) {
  const mode = theme === 'dark' ? 'dark' : 'light';
  document.documentElement.dataset.theme = mode;
  themeBtns.forEach((btn) => {
    btn.classList.toggle('is-active', btn.dataset.setTheme === mode);
  });
  try {
    window.localStorage.setItem('sim_theme_pref_v2', mode);
  } catch {
    // ignore
  }
}

try {
  const saved = window.localStorage.getItem('sim_theme_pref_v2');
  applyTheme(saved || 'light');
} catch {
  applyTheme('light');
}

themeBtns.forEach((btn) => {
  btn.addEventListener('click', () => {
    applyTheme(btn.dataset.setTheme);
    showToast(btn.dataset.setTheme === 'light' ? '화이트 모드로 전환했습니다.' : '다크 모드로 전환했습니다.', 'info');
  });
});

titleEl.textContent = APP_CONFIG.lessonTitle;

function updateHeaderStatus(state) {
  const mod = state.modules.m5;
  const pred = mod.prediction?.choiceId ? 1 : 0;
  const missions = Object.values(mod.missions || {}).filter(Boolean).length;
  const checks = Object.values(mod.checks || {}).filter((c) => c?.correct).length;
  const doneParts = pred + (missions >= 1 ? 1 : 0) + (checks >= 2 ? 1 : 0);

  progressBadgeEl.textContent = `진행 ${doneParts}/3`;
  progressBadgeEl.classList.toggle('is-complete', doneParts === 3);

  if (state.lastSubmission?.updatedAt) {
    submitStatusEl.textContent = `제출 완료 (${state.lastSubmission.studentId} · ${formatKoreaTime(
      state.lastSubmission.updatedAt
    )})`;
    submitStatusEl.classList.add('is-submitted');
  } else {
    submitStatusEl.textContent = '자동 저장 중';
    submitStatusEl.classList.remove('is-submitted');
  }
}

store.subscribe(updateHeaderStatus);
updateHeaderStatus(store.state);

// 상단 스크롤 진행 게이지
function updateScrollProgress() {
  if (!scrollBarEl) return;
  const maxScroll = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
  const pct = Math.min(100, Math.max(0, (window.scrollY / maxScroll) * 100));
  scrollBarEl.style.width = `${pct.toFixed(1)}%`;
}
window.addEventListener('scroll', updateScrollProgress, { passive: true });
updateScrollProgress();

// 제출 모달
const submitModal = initSubmitModal(store, () => {
  updateHeaderStatus(store.state);
});
openSubmitBtn.addEventListener('click', submitModal.openModal);

// M5 모듈 마운트
const m5Instance = mountM5Module({
  rootEl: document.getElementById('app-root'),
  store,
  onOpenSubmit: submitModal.openModal,
});

// 발표 모드 토글
let presentationOn = false;
presentBtn.addEventListener('click', () => {
  presentationOn = !presentationOn;
  document.body.classList.toggle('is-presentation', presentationOn);
  presentBtn.classList.toggle('is-active', presentationOn);
  presentBtn.textContent = presentationOn ? '발표 모드 ON' : '발표 모드';
  m5Instance.setPresentationMode(presentationOn);
  showToast(
    presentationOn
      ? '발표 모드 ON: 글자 확대 · 시연 프리셋 활성화 · 확인 문제 숨김'
      : '일반 학생 모드로 돌아왔습니다.',
    'info'
  );
});
