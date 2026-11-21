/**
 * M5. 역전파 한 걸음씩 — 초대형 와이드 시뮬레이션 + 키워드 중심 슬라이드 덱 (v5)
 * - 1600×540 초대형 글래스/뉴모피즘 신경망 캔버스를 화면 중앙 풀와이드(96vw)로 배치
 * - 6대 핵심 키워드 마스터리 바: 클릭 즉시 캔버스 애니메이션 발사 + 해당 슬라이드 이동
 * - 좌우 슬라이드 캐러셀(1~8번 슬라이드, 버튼·키보드 ←/→·마우스 스와이프 지원) + 키워드 카드 학습 구조
 */
import {
  DEFAULT_PARAMS,
  cloneMatrix,
  fmt,
  simulateUpdate,
  stepFullBackprop,
} from '../math/backprop.js';
import { M5_CONTENT } from '../../content/m5-backprop.js';
import { renderKeywordWidget, renderMcqWidget, renderPredictionWidget } from '../core/quiz.js';
import { flashElement, showToast } from '../core/ui.js';
import { mountEntropyCanvas } from '../visuals/entropy-bg.js';
import { createNeuralCanvas } from '../visuals/neural-canvas.js';
import {
  renderChapter1Diagram,
  renderChapter2Diagram,
  renderChapter3Diagram,
  renderChapter4Diagram,
  renderChapter5Diagram,
} from '../visuals/chapter-diagrams.js';
import { buildInspectorHtml } from '../visuals/inspector.js';

const KEYWORD_DECK = [
  { id: 'kw_forward', label: '#순전파', sub: '입력→은닉→출력', slideIdx: 1, stage: 1, focusNode: 'h1' },
  { id: 'kw_loss', label: '#제곱오차', sub: '0.5×(목표−출력)²', slideIdx: 2, stage: 2, focusNode: 'o1' },
  { id: 'kw_step1', label: '#오차방향', sub: '1단계 −(목표−출력)', slideIdx: 3, stage: 3, focusNode: 'o1' },
  { id: 'kw_step2', label: '#시그모이드민감도', sub: '2단계 출력×(1−출력)', slideIdx: 4, stage: 4, focusNode: 'o1' },
  { id: 'kw_step3', label: '#입력신호책임', sub: '3단계 앞 은닉값 h', slideIdx: 5, stage: 5, focusNode: 'h1' },
  { id: 'kw_chain', label: '#체인룰·학습률', sub: '세 값의 곱 업데이트', slideIdx: 6, stage: 6, focusNode: 'o1' },
];

const SLIDE_META = [
  { idx: 0, tab: '1. 예상하기', stage: 1 },
  { idx: 1, tab: '2. #순전파', stage: 1 },
  { idx: 2, tab: '3. #제곱오차', stage: 2 },
  { idx: 3, tab: '4. #오차방향 ①', stage: 3 },
  { idx: 4, tab: '5. #민감도 ②', stage: 4 },
  { idx: 5, tab: '6. #입력책임 ③', stage: 5 },
  { idx: 6, tab: '7. #체인룰 실험실', stage: 6 },
  { idx: 7, tab: '8. 퀴즈 · 제출', stage: 6 },
];

function createLocalState() {
  return {
    inputs: [...DEFAULT_PARAMS.inputs],
    targets: [...DEFAULT_PARAMS.targets],
    W1: cloneMatrix(DEFAULT_PARAMS.W1),
    W2: cloneMatrix(DEFAULT_PARAMS.W2),
    lr: DEFAULT_PARAMS.lr,
    selectedWeight: 'w5',
    inspectedId: null,
    loupeOn: false,
    theaterOn: false,
    slideIdx: 0, // 0..7
    stage: 1, // 1..6
    flowPhase: 'auto',
    masteredKeywords: new Set(['kw_forward']),
    appliedUpdates: 0,
    errorHistory: [0.245532],
    isLiveTraining: false,
    liveTrainTimer: null,
    liveTick: 0,
    presentationMode: false,
  };
}

export function mountM5Module({ rootEl, store, onOpenSubmit }) {
  const local = createLocalState();
  const savedMod = store.getModuleState('m5');

  if (savedMod.prediction?.choiceId) {
    local.slideIdx = 1;
  }

  rootEl.innerHTML = `
    <!-- 페이지 배경 앰비언트 글래스 오로라 오브 -->
    <div class="ambient-mesh" aria-hidden="true">
      <div class="ambient-orb is-orb-1"></div>
      <div class="ambient-orb is-orb-2"></div>
      <div class="ambient-orb is-orb-3"></div>
    </div>

    <!-- 컴팩트 히어로 배너 + 키워드 마스터리 트래커 -->
    <section class="hero-compact" id="sec-hero">
      <canvas class="hero-canvas" id="m5-entropy-canvas" aria-hidden="true"></canvas>
      <div class="hero-compact-inner glass-panel">
        <div class="hero-title-group">
          <div class="hero-eyebrow-row">
            <span class="hero-pill-badge">${M5_CONTENT.badge}</span>
            <span class="kw-mastery-counter" id="m5-kw-counter">핵심 키워드 1 / 6 탐험</span>
          </div>
          <h1 class="hero-main-title">
            출력의 <span class="text-back">#오차</span>는 어떻게
            <span class="text-fwd">#체인룰(세 값의 곱)</span>로 가중치 <span class="text-fwd">w5</span>를 고칠까?
          </h1>
        </div>

        <!-- 클릭형 핵심 키워드 6개 마스터리 바 -->
        <div class="keyword-mastery-bar" id="m5-kw-bar" role="navigation" aria-label="핵심 학습 키워드 바로가기"></div>
      </div>
    </section>

    <!-- 초대형 풀와이드(96vw) 시뮬레이션 아레나 + 하단 슬라이드 덱 -->
    <section class="arena-section" id="sec-scrolly">
      <div class="arena-container" id="m5-arena-container">
        <!-- 1. 초대형 시뮬레이션 스튜디오 (글래스모피즘 + 뉴모피즘) -->
        <div class="neural-studio glass-panel" id="m5-sticky-studio">
          <div class="studio-header">
            <div class="weight-selector" role="group" aria-label="추적할 가중치 선택">
              <span class="selector-label">추적 가중치</span>
              <button type="button" class="neu-pill is-active" data-weight="w5">w5 <small>h1→o1</small></button>
              <button type="button" class="neu-pill" data-weight="w6">w6 <small>h1→o2</small></button>
              <button type="button" class="neu-pill" data-weight="w7">w7 <small>h2→o1</small></button>
              <button type="button" class="neu-pill" data-weight="w8">w8 <small>h2→o2</small></button>
            </div>

            <div class="stage-pills" id="m5-stage-pills" role="tablist" aria-label="시뮬레이션 6단계"></div>

            <div class="mode-switch-group" role="group" aria-label="시뮬레이션 도구">
              <button type="button" class="tool-chip" id="m5-loupe-btn" title="마우스 주변 2.2배 확대 + 내부 값 투시 (단축키 Z)">🔍 돋보기 (Z)</button>
              <div class="zoom-group" role="group" aria-label="캔버스 확대 축소">
                <button type="button" class="zoom-btn" id="m5-zoom-out" aria-label="축소">−</button>
                <button type="button" class="zoom-btn is-label" id="m5-zoom-reset">100%</button>
                <button type="button" class="zoom-btn" id="m5-zoom-in" aria-label="확대">+</button>
              </div>
              <button type="button" class="tool-chip" id="m5-theater-btn" title="시뮬레이션 화면 최대 확대">⛶ 크게 보기</button>
              <button type="button" class="mode-chip" id="m5-mode-fwd" data-mode="forward">순전파 →</button>
              <button type="button" class="mode-chip" id="m5-mode-back" data-mode="backward">← 역전파</button>
            </div>
          </div>

          <div class="preset-bar" id="m5-preset-bar" hidden>
            <span class="selector-label">교사 시연 프리셋</span>
            <div class="preset-list" id="m5-preset-list"></div>
          </div>

          <!-- 1600×540 초대형 라이브 신경망 캔버스 -->
          <div class="canvas-stage" id="m5-canvas-stage">
            <canvas id="m5-neural-canvas" aria-label="실시간 2-2-2 신경망 순전파 및 역전파 시뮬레이션"></canvas>
            <div class="canvas-tooltip" id="m5-canvas-tooltip" hidden></div>

            <!-- 클릭 부연설명 인스펙터 드로어 -->
            <aside class="inspector-drawer glass-panel" id="m5-inspector-drawer" hidden aria-live="polite">
              <div class="insp-header">
                <div>
                  <span class="insp-badge" id="m5-insp-badge">-</span>
                  <h4 class="insp-title" id="m5-insp-title">-</h4>
                </div>
                <button type="button" class="btn btn-ghost btn-sm" id="m5-insp-close" aria-label="해설 닫기">✕</button>
              </div>
              <div class="insp-body" id="m5-insp-body"></div>
            </aside>
          </div>

          <!-- 박진감 있는 3단계 체인 이퀄라이저 + 즉시 실행 액션 바 -->
          <div class="studio-action-deck">
            <div class="chain-board" id="m5-chain-board">
              <button type="button" class="chain-card neu-card" data-chain-step="3" title="클릭해 1단계 오차방향 보기">
                <div class="chain-top">
                  <span class="chain-tag">1단계 · #오차방향</span>
                  <span class="chain-formula">−(목표−출력)</span>
                </div>
                <div class="chain-val num" id="m5-val-s1">-</div>
                <div class="chain-meter"><div class="chain-meter-fill is-back" id="m5-bar-s1"></div></div>
                <div class="chain-desc" id="m5-desc-s1">출력 > 목표</div>
              </button>

              <div class="chain-op">×</div>

              <button type="button" class="chain-card neu-card" data-chain-step="4" title="클릭해 2단계 시그모이드 민감도 보기">
                <div class="chain-top">
                  <span class="chain-tag">2단계 · #민감도</span>
                  <span class="chain-formula">출력×(1−출력)</span>
                </div>
                <div class="chain-val num" id="m5-val-s2">-</div>
                <div class="chain-meter"><div class="chain-meter-fill is-back" id="m5-bar-s2"></div></div>
                <div class="chain-desc" id="m5-desc-s2">최대 0.25</div>
              </button>

              <div class="chain-op">×</div>

              <button type="button" class="chain-card neu-card" data-chain-step="5" title="클릭해 3단계 들어온 신호 보기">
                <div class="chain-top">
                  <span class="chain-tag">3단계 · #입력신호</span>
                  <span class="chain-formula">앞 은닉값 h</span>
                </div>
                <div class="chain-val num" id="m5-val-s3">-</div>
                <div class="chain-meter"><div class="chain-meter-fill is-fwd" id="m5-bar-s3"></div></div>
                <div class="chain-desc" id="m5-desc-s3">h1 신호</div>
              </button>

              <div class="chain-op">=</div>

              <button type="button" class="chain-card neu-card is-result" data-chain-step="6" title="클릭해 가중치 업데이트 보기">
                <div class="chain-top">
                  <span class="chain-tag">#가중치영향력 (기울기)</span>
                  <span class="chain-formula">∂E / ∂w</span>
                </div>
                <div class="chain-val num" id="m5-val-grad">-</div>
                <div class="chain-meter"><div class="chain-meter-fill is-result" id="m5-bar-grad"></div></div>
                <div class="chain-desc" id="m5-desc-grad">세 값의 곱</div>
              </button>
            </div>

            <!-- 우측 즉시 발사 콘솔 & 실시간 오차 스파크라인 -->
            <div class="quick-fire-console neu-card">
              <div class="fire-preview-row">
                <div class="hud-metric">
                  <span class="metric-label" id="m5-weight-change-label">w5 수정 미리보기</span>
                  <div class="metric-row" id="m5-weight-change-row"></div>
                </div>
                <div class="hud-sparkline">
                  <div class="sparkline-head">
                    <span class="metric-label">오차 추이 (<span id="m5-update-counter">0회</span>)</span>
                    <strong class="num spark-val" id="m5-spark-val">0.245532</strong>
                  </div>
                  <svg id="m5-sparkline-svg" viewBox="0 0 200 34" aria-label="오차 변화 그래프"></svg>
                </div>
              </div>
              <div class="fire-btn-row">
                <button type="button" class="btn btn-primary btn-sm" id="m5-apply-one">⚡ w5 1회 수정 발사</button>
                <button type="button" class="btn btn-outline btn-sm" id="m5-apply-all">🔥 w5~w8 일괄 수정</button>
                <button type="button" class="btn btn-back btn-sm" id="m5-live-loop">🚀 왕복 연속 학습</button>
                <button type="button" class="btn btn-ghost btn-sm" id="m5-reset-weights">초기화</button>
              </div>
            </div>
          </div>
        </div>

        <!-- 2. 키워드 중심 인터랙티브 슬라이드 캐러셀 (슬라이드 1~8) -->
        <div class="slide-deck-shell glass-panel" id="sec-predict">
          <div class="slide-deck-topbar">
            <div class="slide-tabs" id="m5-slide-tabs" role="tablist" aria-label="학습 슬라이드 목록"></div>
            <div class="slide-nav-controls">
              <button type="button" class="btn btn-outline btn-sm" id="m5-prev-slide">← 이전 슬라이드</button>
              <span class="slide-page-indicator num" id="m5-slide-indicator">1 / 8</span>
              <button type="button" class="btn btn-primary btn-sm" id="m5-next-slide">다음 슬라이드 →</button>
            </div>
          </div>

          <div class="slide-viewport" id="m5-slide-viewport">
            <div class="slide-track" id="m5-slide-track">
              <!-- SLIDE 1: 먼저 예상하기 -->
              <article class="slide-card" data-slide-index="0">
                <div class="slide-col-left">
                  <div class="kw-chip-row">
                    <span class="kw-tag is-fwd">#예상하기</span>
                    <span class="kw-tag is-back">#목표와출력비교</span>
                    <span class="kw-tag">#가중치수정방향</span>
                  </div>
                  <h3>STEP 01 · 조작하기 전에 결과를 먼저 짐작해 봅시다</h3>
                  <div class="kw-flow-grid">
                    <div class="kw-flow-box">
                      <small>현재 출력 o1</small>
                      <strong class="num text-fwd">0.6005</strong>
                      <span>너무 크게 나옴</span>
                    </div>
                    <div class="kw-flow-arrow">vs</div>
                    <div class="kw-flow-box">
                      <small>목표 target_o1</small>
                      <strong class="num">0.0100</strong>
                      <span>낮춰야 할 목표</span>
                    </div>
                    <div class="kw-flow-arrow">→</div>
                    <div class="kw-flow-box is-highlight">
                      <small>연결 가중치 w5</small>
                      <strong class="num text-back">0.3500 → ?</strong>
                      <span>커질까, 작아질까?</span>
                    </div>
                  </div>
                  <p class="slide-helper-text">
                    위 초대형 신경망에서 <strong>h1 → o1</strong>을 잇는 <strong>w5(0.3500)</strong>를 보세요. 오른쪽에서 예상을 하나 고르면 다음 슬라이드로 넘어갑니다.
                  </p>
                </div>
                <div class="slide-col-right neu-inset-box">
                  <div id="m5-prediction-mount"></div>
                </div>
              </article>

              <!-- SLIDE 2: 순전파 -->
              <article class="slide-card" data-slide-index="1">
                <div class="slide-col-left">
                  <div class="kw-chip-row">
                    <button type="button" class="kw-tag is-fwd is-clickable" data-inspect-target="h1">#순전파 (클릭해 h1 해부)</button>
                    <button type="button" class="kw-tag is-fwd is-clickable" data-inspect-target="w5">#가중합 (w5 해부)</button>
                    <button type="button" class="kw-tag is-fwd is-clickable" data-inspect-target="o1">#시그모이드 (o1 해부)</button>
                  </div>
                  <h3>01 · #순전파 — 입력 신호가 앞으로 흘러 예측값을 만든다</h3>
                  <div class="kw-flow-grid">
                    <div class="kw-flow-box">
                      <small>1. 곱해서 더하기</small>
                      <strong>#가중합 (net)</strong>
                      <span>입력 × 가중치 합산</span>
                    </div>
                    <div class="kw-flow-arrow">→</div>
                    <div class="kw-flow-box">
                      <small>2. 0~1로 압축</small>
                      <strong>#시그모이드 σ(x)</strong>
                      <span>은닉 h1, h2 생성</span>
                    </div>
                    <div class="kw-flow-arrow">→</div>
                    <div class="kw-flow-box is-highlight">
                      <small>3. 최종 예측</small>
                      <strong class="text-fwd">o1, o2 출력</strong>
                      <span>파란 입자 흐름 →</span>
                    </div>
                  </div>
                  <p class="slide-helper-text">
                    위 키워드 태그나 신경망의 <strong>h1, o1 노드</strong>를 직접 클릭해 보세요. 가중합과 시그모이드 통과 과정이 숫자로 즉시 해부됩니다.
                  </p>
                </div>
                <div class="slide-col-right neu-inset-box" id="m5-ch1-diagram"></div>
              </article>

              <!-- SLIDE 3: 오차 계산 -->
              <article class="slide-card" data-slide-index="2">
                <div class="slide-col-left">
                  <div class="kw-chip-row">
                    <button type="button" class="kw-tag is-back is-clickable" data-inspect-target="E_total">#손실함수 (E_total 해부)</button>
                    <button type="button" class="kw-tag is-back is-clickable" data-inspect-target="E_o1">#제곱오차 (E_o1 해부)</button>
                    <button type="button" class="kw-tag is-clickable" data-inspect-target="E_o2">#0.5를곱하는이유</button>
                  </div>
                  <h3>02 · #제곱오차 — 예측이 목표에서 얼마나 벗어났는지 잰다</h3>
                  <div class="kw-flow-grid">
                    <div class="kw-flow-box">
                      <small>1. 차이 구하기</small>
                      <strong>목표 − 출력</strong>
                      <span>o1은 −0.59, o2는 +0.38</span>
                    </div>
                    <div class="kw-flow-arrow">→</div>
                    <div class="kw-flow-box">
                      <small>2. 부호 없애기</small>
                      <strong>#차이제곱 × 0.5</strong>
                      <span>미분 시 2와 약분됨</span>
                    </div>
                    <div class="kw-flow-arrow">→</div>
                    <div class="kw-flow-box is-highlight">
                      <small>3. 전체 오차 합산</small>
                      <strong class="num text-back">E = 0.245532</strong>
                      <span>0에 가깝게 줄일 대상</span>
                    </div>
                  </div>
                  <p class="slide-helper-text">
                    차이를 제곱하면 부호(+/−)가 사라지고 크게 틀릴수록 벌점이 커집니다. 앞에 <strong>0.5</strong>를 곱해 두면 미분할 때 식이 깔끔해집니다.
                  </p>
                </div>
                <div class="slide-col-right neu-inset-box" id="m5-ch2-diagram"></div>
              </article>

              <!-- SLIDE 4: 역전파 1단계 -->
              <article class="slide-card" data-slide-index="3">
                <div class="slide-col-left">
                  <div class="kw-chip-row">
                    <span class="kw-tag is-back">#역전파1단계</span>
                    <span class="kw-tag is-back">#오차방향</span>
                    <button type="button" class="kw-tag is-clickable" data-quick-weight="w5">w5 보기 (+0.5905)</button>
                    <button type="button" class="kw-tag is-clickable" data-quick-weight="w6">w6 비교 (−0.3774)</button>
                  </div>
                  <h3>03 · #오차방향 (1단계) — 가중치를 줄일지 키울지 부호로 결정한다</h3>
                  <div class="kw-flow-grid">
                    <div class="kw-flow-box">
                      <small>출력 > 목표 (o1)</small>
                      <strong class="text-back">양수 (+) 신호</strong>
                      <span>가중치를 빼서 줄임 ↓</span>
                    </div>
                    <div class="kw-flow-arrow">↔</div>
                    <div class="kw-flow-box">
                      <small>출력 < 목표 (o2)</small>
                      <strong class="text-fwd">음수 (−) 신호</strong>
                      <span>마이너스×마이너스=증가 ↑</span>
                    </div>
                    <div class="kw-flow-arrow">→</div>
                    <div class="kw-flow-box is-highlight">
                      <small>1단계 공식</small>
                      <strong class="num">−(목표 − 출력)</strong>
                      <span>= 출력 − 목표</span>
                    </div>
                  </div>
                  <p class="slide-helper-text">
                    위 <strong>[w6 비교 (−0.3774)]</strong> 태그를 눌러 보세요! o2(0.61)는 목표(0.99)보다 작아서 1단계 값이 음수(−)가 되고, 가중치 w6은 반대로 커집니다(미션 2 달성!).
                  </p>
                </div>
                <div class="slide-col-right neu-inset-box" id="m5-ch3-diagram"></div>
              </article>

              <!-- SLIDE 5: 역전파 2단계 -->
              <article class="slide-card" data-slide-index="4">
                <div class="slide-col-left">
                  <div class="kw-chip-row">
                    <span class="kw-tag is-back">#역전파2단계</span>
                    <span class="kw-tag is-back">#시그모이드민감도</span>
                    <span class="kw-tag">#최대기울기0.25</span>
                    <span class="kw-tag">#기울기소실원인</span>
                  </div>
                  <h3>04 · #시그모이드민감도 (2단계) — 출력 노드가 얼마나 잘 변할 수 있는가</h3>
                  <div class="kw-flow-grid">
                    <div class="kw-flow-box is-highlight">
                      <small>출력 ≈ 0.5 부근</small>
                      <strong class="text-back">최대 민감 (0.25)</strong>
                      <span>가파른 비탈길</span>
                    </div>
                    <div class="kw-flow-arrow">vs</div>
                    <div class="kw-flow-box">
                      <small>출력 → 0 또는 1</small>
                      <strong>둔감 (기울기 → 0)</strong>
                      <span>평평한 고원 구간</span>
                    </div>
                    <div class="kw-flow-arrow">→</div>
                    <div class="kw-flow-box">
                      <small>2단계 공식</small>
                      <strong class="num">출력 × (1 − 출력)</strong>
                      <span>시그모이드 도함수</span>
                    </div>
                  </div>
                  <p class="slide-helper-text">
                    오른쪽 그래프의 주황색 접선을 보세요. 현재 <strong>o1(0.6005)</strong>은 0.5 근처라 기울기가 <strong>0.239910</strong>(최대 0.25의 96%)으로 매우 활발하게 학습됩니다.
                  </p>
                </div>
                <div class="slide-col-right neu-inset-box" id="m5-ch4-diagram"></div>
              </article>

              <!-- SLIDE 6: 역전파 3단계 -->
              <article class="slide-card" data-slide-index="5">
                <div class="slide-col-left">
                  <div class="kw-chip-row">
                    <span class="kw-tag is-fwd">#역전파3단계</span>
                    <span class="kw-tag is-fwd">#입력신호책임</span>
                    <button type="button" class="kw-tag is-clickable" data-quick-weight="w5">w5·w6 → h1 신호</button>
                    <button type="button" class="kw-tag is-clickable" data-quick-weight="w7">w7·w8 → h2 신호</button>
                  </div>
                  <h3>05 · #입력신호책임 (3단계) — 이 선으로 들어온 신호가 클수록 책임도 크다</h3>
                  <div class="kw-flow-grid">
                    <div class="kw-flow-box">
                      <small>앞쪽에서 보낸 값</small>
                      <strong class="text-fwd">은닉 신호 h1, h2</strong>
                      <span>파란 순전파 신호 →</span>
                    </div>
                    <div class="kw-flow-arrow">×</div>
                    <div class="kw-flow-box">
                      <small>뒤에서 돌아온 값</small>
                      <strong class="text-back">1단계 × 2단계</strong>
                      <span>← 주황 역전파 오차</span>
                    </div>
                    <div class="kw-flow-arrow">=</div>
                    <div class="kw-flow-box is-highlight">
                      <small>가중치 중앙 합류!</small>
                      <strong class="num">∂E / ∂w 기울기</strong>
                      <span>세 값의 곱 완성</span>
                    </div>
                  </div>
                  <p class="slide-helper-text">
                    위 신경망 캔버스에서 <strong>파란 신호(h1 →)</strong>와 <strong>주황 오차(← o1)</strong>가 <strong>w5 한가운데서 부딪혀 스파크</strong>를 일으키는 장면을 확인하세요!
                  </p>
                </div>
                <div class="slide-col-right neu-inset-box" id="m5-ch5-diagram"></div>
              </article>

              <!-- SLIDE 7: 체인룰 실험실 & 미션 -->
              <article class="slide-card" data-slide-index="6">
                <div class="slide-col-left">
                  <div class="kw-chip-row">
                    <span class="kw-tag is-back">#체인룰(연쇄법칙)</span>
                    <span class="kw-tag is-fwd">#학습률(lr)</span>
                    <span class="kw-tag">#경사하강법</span>
                  </div>
                  <h3>06 · #체인룰 실험실 — 직접 변수를 바꾸고 미션을 달성하세요</h3>
                  <div class="error-compare-box" id="m5-error-compare"></div>
                  <div class="mission-block">
                    <div class="mission-list" id="m5-mission-list"></div>
                  </div>
                  <details class="formula-details">
                    <summary>편미분 체인룰 수식으로 보기 (선택 · 노트북 셀 59~67)</summary>
                    <div class="formula-body" id="m5-formula-body"></div>
                  </details>
                </div>

                <div class="slide-col-right neu-inset-box">
                  <div class="controls-header">
                    <h4>실시간 하이퍼파라미터 · 입력 조절</h4>
                    <button type="button" class="btn btn-ghost btn-sm" id="m5-reset-all">기본값 복원</button>
                  </div>
                  <div class="sliders-grid">
                    <label class="slider-item">
                      <div class="slider-label-row">
                        <span>입력 i<sub>1</sub></span>
                        <strong class="num" id="m5-lbl-i1">0.05</strong>
                      </div>
                      <input type="range" id="m5-sl-i1" min="0" max="1" step="0.05" value="0.05" />
                    </label>
                    <label class="slider-item">
                      <div class="slider-label-row">
                        <span>입력 i<sub>2</sub></span>
                        <strong class="num" id="m5-lbl-i2">0.10</strong>
                      </div>
                      <input type="range" id="m5-sl-i2" min="0" max="1" step="0.05" value="0.10" />
                    </label>
                    <label class="slider-item">
                      <div class="slider-label-row">
                        <span>목표 target<sub>o1</sub></span>
                        <strong class="num" id="m5-lbl-t1">0.01</strong>
                      </div>
                      <input type="range" id="m5-sl-t1" min="0.01" max="0.99" step="0.01" value="0.01" />
                    </label>
                    <label class="slider-item">
                      <div class="slider-label-row">
                        <span>#학습률 (lr)</span>
                        <strong class="num" id="m5-lbl-lr">0.05</strong>
                      </div>
                      <input type="range" id="m5-sl-lr" min="0.01" max="1.00" step="0.01" value="0.05" />
                    </label>
                  </div>
                  <p class="slide-helper-text" style="margin-top: 12px">
                    학습률(lr)을 <strong>0.80 이상</strong>으로 키운 뒤 위쪽 <strong>[⚡ w5 1회 수정 발사]</strong>를 눌러 오차가 얼마나 크게 변하는지 실험해 보세요.
                  </p>
                </div>
              </article>

              <!-- SLIDE 8: 확인 퀴즈 & 기록 제출 -->
              <article class="slide-card" data-slide-index="7" id="sec-check">
                <div class="slide-col-left checks-section" id="m5-checks-section">
                  <div class="kw-chip-row">
                    <span class="kw-tag is-fwd">#핵심개념체크</span>
                    <span class="kw-tag is-back">#체인룰완성</span>
                  </div>
                  <h3>07 · 키워드 이해 확인 &amp; 학습 기록 제출</h3>
                  <div class="quiz-card neu-inset-box" id="m5-check-1"></div>
                  <div class="quiz-card neu-inset-box" id="m5-check-2"></div>
                </div>

                <div class="slide-col-right" style="display:flex;flex-direction:column;gap:12px">
                  <div class="prediction-compare-card neu-inset-box" id="m5-pred-compare" hidden></div>

                  <div class="reflection-card neu-inset-box">
                    <label for="m5-reflection-input" class="quiz-question">${M5_CONTENT.reflection.question}</label>
                    <textarea
                      id="m5-reflection-input"
                      class="text-area"
                      rows="2"
                      placeholder="${M5_CONTENT.reflection.placeholder}"
                    ></textarea>
                    <div class="reflection-meta">${M5_CONTENT.reflection.notebookRef} · 자동 저장</div>
                  </div>

                  <div class="notebook-card neu-inset-box">
                    <h4>실습 노트북 연결 (3장 셀 53~71)</h4>
                    <ul>
                      ${M5_CONTENT.notebookLinks
                        .map((item) => `<li><strong>${item.cells}</strong> — ${item.desc}</li>`)
                        .join('')}
                    </ul>
                  </div>

                  <div class="final-submit-banner neu-card">
                    <div>
                      <h4>학습 기록 제출</h4>
                      <p>학번 · 이름 · 비밀번호를 입력해 오늘 실습 기록을 저장합니다.</p>
                    </div>
                    <button type="button" class="btn btn-primary" id="m5-panel-submit-btn">
                      기록 제출하기 →
                    </button>
                  </div>
                </div>
              </article>
            </div>
          </div>
        </div>
      </div>
    </section>
  `;

  // ===== 1. 히어로 파티클 & 1600×540 초대형 신경망 캔버스 마운트 =====
  const destroyEntropy = mountEntropyCanvas(rootEl.querySelector('#m5-entropy-canvas'));
  const neuralCanvas = createNeuralCanvas(
    rootEl.querySelector('#m5-neural-canvas'),
    rootEl.querySelector('#m5-canvas-tooltip'),
    {
      onSelectWeight: (wKey) => selectWeight(wKey),
      onInspectElement: (targetId) => openInspector(targetId),
    }
  );

  // ===== 2. 핵심 키워드 6개 마스터리 바 생성 =====
  const kwBarEl = rootEl.querySelector('#m5-kw-bar');
  const kwCounterEl = rootEl.querySelector('#m5-kw-counter');

  function renderKeywordBar() {
    kwCounterEl.textContent = `핵심 키워드 ${local.masteredKeywords.size} / ${KEYWORD_DECK.length} 탐험`;
    kwBarEl.innerHTML = KEYWORD_DECK.map((kw) => {
      const done = local.masteredKeywords.has(kw.id);
      const active = local.slideIdx === kw.slideIdx;
      return `
        <button type="button" class="kw-mastery-pill ${done ? 'is-done' : ''} ${active ? 'is-active' : ''}" data-kw-id="${kw.id}">
          <span class="kw-check">${done ? '✓' : '●'}</span>
          <span class="kw-main">${kw.label}</span>
          <small class="kw-sub">${kw.sub}</small>
        </button>
      `;
    }).join('');

    kwBarEl.querySelectorAll('[data-kw-id]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const kw = KEYWORD_DECK.find((k) => k.id === btn.dataset.kwId);
        if (!kw) return;
        stopLiveTraining();
        local.masteredKeywords.add(kw.id);
        local.flowPhase = 'auto';
        goToSlide(kw.slideIdx);
        neuralCanvas.triggerRipple(kw.focusNode);
        neuralCanvas.spawnFloatingText(
          kw.focusNode === 'h1' ? 555 : 980,
          105,
          `${kw.label} 집중!`,
          kw.stage >= 3 ? '#f97316' : '#3b82f6'
        );
      });
    });
  }

  // ===== 3. DOM 참조 & 슬라이드 탭 생성 =====
  const arenaContainerEl = rootEl.querySelector('#m5-arena-container');
  const stagePillsEl = rootEl.querySelector('#m5-stage-pills');
  const slideTabsEl = rootEl.querySelector('#m5-slide-tabs');
  const slideTrackEl = rootEl.querySelector('#m5-slide-track');
  const slideViewportEl = rootEl.querySelector('#m5-slide-viewport');
  const slideCards = Array.from(slideTrackEl.querySelectorAll('.slide-card'));
  const slideIndicatorEl = rootEl.querySelector('#m5-slide-indicator');

  function syncSlideViewportHeight() {
    const activeCard = slideCards[local.slideIdx];
    if (!activeCard || !slideViewportEl) return;
    const h = Math.ceil(activeCard.offsetHeight);
    if (h > 0) slideViewportEl.style.height = `${h}px`;
  }

  let slideResizeObserver = null;
  if (typeof ResizeObserver !== 'undefined') {
    slideResizeObserver = new ResizeObserver(() => {
      syncSlideViewportHeight();
    });
    slideCards.forEach((card) => slideResizeObserver.observe(card));
  }
  const prevSlideBtn = rootEl.querySelector('#m5-prev-slide');
  const nextSlideBtn = rootEl.querySelector('#m5-next-slide');

  const updateCounterEl = rootEl.querySelector('#m5-update-counter');
  const modeFwdBtn = rootEl.querySelector('#m5-mode-fwd');
  const modeBackBtn = rootEl.querySelector('#m5-mode-back');
  const loupeBtn = rootEl.querySelector('#m5-loupe-btn');
  const theaterBtn = rootEl.querySelector('#m5-theater-btn');
  const zoomInBtn = rootEl.querySelector('#m5-zoom-in');
  const zoomOutBtn = rootEl.querySelector('#m5-zoom-out');
  const zoomResetBtn = rootEl.querySelector('#m5-zoom-reset');
  const inspDrawerEl = rootEl.querySelector('#m5-inspector-drawer');
  const inspBadgeEl = rootEl.querySelector('#m5-insp-badge');
  const inspTitleEl = rootEl.querySelector('#m5-insp-title');
  const inspBodyEl = rootEl.querySelector('#m5-insp-body');
  const inspCloseBtn = rootEl.querySelector('#m5-insp-close');
  const presetBarEl = rootEl.querySelector('#m5-preset-bar');
  const presetListEl = rootEl.querySelector('#m5-preset-list');

  const valS1 = rootEl.querySelector('#m5-val-s1');
  const valS2 = rootEl.querySelector('#m5-val-s2');
  const valS3 = rootEl.querySelector('#m5-val-s3');
  const valGrad = rootEl.querySelector('#m5-val-grad');
  const barS1 = rootEl.querySelector('#m5-bar-s1');
  const barS2 = rootEl.querySelector('#m5-bar-s2');
  const barS3 = rootEl.querySelector('#m5-bar-s3');
  const barGrad = rootEl.querySelector('#m5-bar-grad');
  const descS1 = rootEl.querySelector('#m5-desc-s1');
  const descS2 = rootEl.querySelector('#m5-desc-s2');
  const descS3 = rootEl.querySelector('#m5-desc-s3');
  const descGrad = rootEl.querySelector('#m5-desc-grad');
  const weightChangeLabel = rootEl.querySelector('#m5-weight-change-label');
  const weightChangeRow = rootEl.querySelector('#m5-weight-change-row');
  const sparkValEl = rootEl.querySelector('#m5-spark-val');
  const sparkSvgEl = rootEl.querySelector('#m5-sparkline-svg');
  const errorCompareEl = rootEl.querySelector('#m5-error-compare');
  const formulaBodyEl = rootEl.querySelector('#m5-formula-body');
  const applyOneBtn = rootEl.querySelector('#m5-apply-one');
  const liveLoopBtn = rootEl.querySelector('#m5-live-loop');

  const slI1 = rootEl.querySelector('#m5-sl-i1');
  const slI2 = rootEl.querySelector('#m5-sl-i2');
  const slT1 = rootEl.querySelector('#m5-sl-t1');
  const slLr = rootEl.querySelector('#m5-sl-lr');
  const lblI1 = rootEl.querySelector('#m5-lbl-i1');
  const lblI2 = rootEl.querySelector('#m5-lbl-i2');
  const lblT1 = rootEl.querySelector('#m5-lbl-t1');
  const lblLr = rootEl.querySelector('#m5-lbl-lr');

  // 상단 1~6단계 필 생성 (슬라이드 1~6과 양방향 연동)
  M5_CONTENT.stages.slice(1).forEach((st) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'stage-pill-btn';
    btn.dataset.stagePill = String(st.step);
    btn.textContent = st.shortTitle;
    btn.addEventListener('click', () => {
      stopLiveTraining();
      local.flowPhase = 'auto';
      // stage 1..6 corresponds to slideIdx 1..6
      goToSlide(st.step);
    });
    stagePillsEl.appendChild(btn);
  });

  // 슬라이드 탭 버튼 생성 (1~8)
  SLIDE_META.forEach((sm) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'slide-tab-btn';
    btn.dataset.slideTab = String(sm.idx);
    btn.textContent = sm.tab;
    btn.addEventListener('click', () => {
      stopLiveTraining();
      local.flowPhase = 'auto';
      goToSlide(sm.idx);
    });
    slideTabsEl.appendChild(btn);
  });

  function goToSlide(nextIdx) {
    local.slideIdx = Math.max(0, Math.min(SLIDE_META.length - 1, nextIdx));
    const mappedStage = SLIDE_META[local.slideIdx].stage;
    const matchedKw = KEYWORD_DECK.find((k) => k.slideIdx === local.slideIdx);
    if (matchedKw) local.masteredKeywords.add(matchedKw.id);
    setStage(mappedStage);
  }

  prevSlideBtn.addEventListener('click', () => {
    stopLiveTraining();
    goToSlide(local.slideIdx - 1);
  });
  nextSlideBtn.addEventListener('click', () => {
    stopLiveTraining();
    goToSlide(local.slideIdx + 1);
  });

  // 슬라이드 트랙 터치/마우스 드래그 스와이프 지원
  let touchStartX = null;
  slideViewportEl.addEventListener('pointerdown', (e) => {
    if (['INPUT', 'TEXTAREA', 'BUTTON', 'SUMMARY'].includes(e.target.tagName)) return;
    touchStartX = e.clientX;
  });
  slideViewportEl.addEventListener('pointerup', (e) => {
    if (touchStartX === null) return;
    const diff = e.clientX - touchStartX;
    touchStartX = null;
    if (Math.abs(diff) > 65) {
      if (diff < 0) goToSlide(local.slideIdx + 1);
      else goToSlide(local.slideIdx - 1);
    }
  });

  // 3단계 이퀄라이저 카드 클릭 시 해당 단계 슬라이드로 즉시 점프
  rootEl.querySelectorAll('[data-chain-step]').forEach((card) => {
    card.addEventListener('click', () => {
      const stepNum = Number(card.dataset.chainStep);
      stopLiveTraining();
      local.flowPhase = 'auto';
      goToSlide(stepNum);
    });
  });

  // 슬라이드 내부 키워드 태그 클릭 액션 (인스펙터 열기 또는 가중치 전환)
  rootEl.querySelectorAll('[data-inspect-target]').forEach((btn) => {
    btn.addEventListener('click', () => {
      openInspector(btn.dataset.inspectTarget);
    });
  });
  rootEl.querySelectorAll('[data-quick-weight]').forEach((btn) => {
    btn.addEventListener('click', () => {
      selectWeight(btn.dataset.quickWeight);
      neuralCanvas.spawnFloatingText(760, 120, `${btn.dataset.quickWeight} 선택!`, '#f97316');
    });
  });

  // 돋보기(2.2×) · 극장 모드 · 줌 컨트롤
  function toggleLoupe() {
    local.loupeOn = !local.loupeOn;
    neuralCanvas.setLoupe(local.loupeOn);
    loupeBtn.classList.toggle('is-active', local.loupeOn);
    loupeBtn.textContent = local.loupeOn ? '🔍 돋보기 ON' : '🔍 돋보기 (Z)';
    showToast(
      local.loupeOn
        ? '돋보기 켜짐: 신경망 위에 마우스를 올리면 2.2배 확대 + 내부 계산값을 투시합니다.'
        : '돋보기를 껐습니다.',
      'info'
    );
  }
  loupeBtn.addEventListener('click', toggleLoupe);

  theaterBtn.addEventListener('click', () => {
    local.theaterOn = !local.theaterOn;
    arenaContainerEl.classList.toggle('is-theater', local.theaterOn);
    theaterBtn.classList.toggle('is-active', local.theaterOn);
    theaterBtn.textContent = local.theaterOn ? '⛶ 기본 크기' : '⛶ 크게 보기';
  });

  function updateZoomLabel(z) {
    zoomResetBtn.textContent = `${Math.round(z * 100)}%`;
  }
  zoomInBtn.addEventListener('click', () => updateZoomLabel(neuralCanvas.setZoom(neuralCanvas.getZoom() + 0.25)));
  zoomOutBtn.addEventListener('click', () => updateZoomLabel(neuralCanvas.setZoom(neuralCanvas.getZoom() - 0.25)));
  zoomResetBtn.addEventListener('click', () => updateZoomLabel(neuralCanvas.setZoom(1.0)));

  function openInspector(targetId) {
    local.inspectedId = targetId;
    renderInspector();
    renderAll();
  }

  function closeInspector() {
    local.inspectedId = null;
    inspDrawerEl.hidden = true;
    renderAll();
  }
  inspCloseBtn.addEventListener('click', closeInspector);

  function renderInspector() {
    if (!local.inspectedId) {
      inspDrawerEl.hidden = true;
      return;
    }
    const sim = getSim(false);
    const info = buildInspectorHtml(local.inspectedId, {
      sim,
      W1: local.W1,
      W2: local.W2,
      lr: local.lr,
    });
    if (!info) {
      inspDrawerEl.hidden = true;
      return;
    }
    inspDrawerEl.hidden = false;
    inspBadgeEl.textContent = info.badge;
    inspBadgeEl.className = `insp-badge is-${info.tone}`;
    inspTitleEl.textContent = info.title;
    inspBodyEl.innerHTML = info.body;

    const applyBtn = inspBodyEl.querySelector('[data-insp-apply]');
    if (applyBtn) {
      applyBtn.addEventListener('click', () => {
        const wKey = applyBtn.getAttribute('data-insp-apply');
        local.selectedWeight = wKey;
        applyOneBtn.click();
      });
    }
  }

  function onGlobalKeyDown(e) {
    if (e.target && ['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
    if (e.key === 'z' || e.key === 'Z') {
      toggleLoupe();
    } else if (e.key === 'ArrowRight') {
      stopLiveTraining();
      goToSlide(local.slideIdx + 1);
    } else if (e.key === 'ArrowLeft') {
      stopLiveTraining();
      goToSlide(local.slideIdx - 1);
    } else if (e.key === 'Escape' && local.inspectedId) {
      closeInspector();
    }
  }
  window.addEventListener('keydown', onGlobalKeyDown);

  modeFwdBtn.addEventListener('click', () => {
    stopLiveTraining();
    local.flowPhase = 'forward';
    goToSlide(1);
  });
  modeBackBtn.addEventListener('click', () => {
    stopLiveTraining();
    local.flowPhase = 'backward';
    if (local.slideIdx < 3) goToSlide(3);
    else renderAll();
  });

  // 교사 발표 모드 프리셋
  M5_CONTENT.presets.forEach((preset) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn btn-ghost btn-sm';
    btn.textContent = preset.label;
    btn.title = preset.desc;
    btn.addEventListener('click', () => {
      stopLiveTraining();
      local.inputs = [...preset.inputs];
      local.targets = [...preset.targets];
      local.W1 = cloneMatrix(DEFAULT_PARAMS.W1);
      local.W2 = cloneMatrix(DEFAULT_PARAMS.W2);
      local.lr = preset.lr;
      local.selectedWeight = preset.weight;
      local.stage = preset.step;
      local.slideIdx = preset.step;
      local.flowPhase = 'auto';
      local.appliedUpdates = 0;
      local.errorHistory = [getSim().before.eTotal];
      syncSlidersFromLocal();
      renderAll();
      showToast(`프리셋 적용: ${preset.label}`, 'info');
    });
    presetListEl.appendChild(btn);
  });

  // ===== 4. 상태 헬퍼 =====
  function getSim(updateAll = false) {
    return simulateUpdate(
      {
        inputs: local.inputs,
        targets: local.targets,
        W1: local.W1,
        W2: local.W2,
        lr: local.lr,
      },
      local.selectedWeight,
      updateAll
    );
  }

  function recordInteraction(updater) {
    store.updateModule('m5', updater);
    renderMissionsAndCompare();
  }

  function completeMission(missionId, toastMsg) {
    const mod = store.getModuleState('m5');
    if (mod.missions[missionId]) return;
    recordInteraction((m) => {
      m.missions[missionId] = { completed: true, completedAt: new Date().toISOString() };
    });
    showToast(toastMsg, 'success');
  }

  function setStage(nextStage) {
    local.stage = Math.max(1, Math.min(6, nextStage));
    recordInteraction((mod) => {
      if (!mod.interactions.stepVisited.includes(local.stage)) {
        mod.interactions.stepVisited.push(local.stage);
      }
    });
    if (local.selectedWeight === 'w6' && local.stage >= 3) {
      completeMission(
        'mission_w6_sign',
        '미션 2 달성: w6은 출력이 목표보다 작아 음수(−) 오차 신호로 값이 커집니다.'
      );
    }
    renderAll();
  }

  function selectWeight(weightKey) {
    local.selectedWeight = weightKey;
    recordInteraction((mod) => {
      if (!mod.interactions.weightsInspected.includes(weightKey)) {
        mod.interactions.weightsInspected.push(weightKey);
      }
    });
    if (weightKey === 'w6' && local.stage >= 3) {
      completeMission(
        'mission_w6_sign',
        '미션 2 달성: w6은 출력이 목표보다 작아 음수(−) 오차 신호로 값이 커집니다.'
      );
    }
    renderAll();
  }

  function stopLiveTraining() {
    local.isLiveTraining = false;
    clearInterval(local.liveTrainTimer);
    local.liveTrainTimer = null;
    local.flowPhase = 'auto';
    liveLoopBtn.textContent = '🚀 왕복 연속 학습';
    liveLoopBtn.classList.remove('is-running');
    renderAll();
  }

  function toggleLiveTraining() {
    if (local.isLiveTraining) {
      stopLiveTraining();
      return;
    }
    local.isLiveTraining = true;
    local.stage = 6;
    local.liveTick = 0;
    liveLoopBtn.textContent = '⏸ 연속 학습 멈춤';
    liveLoopBtn.classList.add('is-running');

    local.liveTrainTimer = setInterval(() => {
      local.liveTick += 1;
      if (local.liveTick % 2 === 1) {
        local.flowPhase = 'forward';
        neuralCanvas.triggerRipple('i1', '#3b82f6');
        neuralCanvas.triggerRipple('i2', '#3b82f6');
        renderAll();
      } else {
        local.flowPhase = 'backward';
        const prevW = local.W2[0][0];
        const res = stepFullBackprop(
          {
            inputs: local.inputs,
            targets: local.targets,
            W1: local.W1,
            W2: local.W2,
            lr: local.lr,
          },
          false
        );
        local.W2 = res.nextParams.W2;
        local.appliedUpdates += 1;
        local.errorHistory.push(res.after.eTotal);
        if (local.errorHistory.length > 50) local.errorHistory.shift();

        neuralCanvas.triggerWeightUpdateEffect(
          local.selectedWeight,
          local.W2[0][0] - prevW,
          res.before.eTotal - res.after.eTotal
        );

        if (res.after.eTotal < res.before.eTotal) {
          completeMission(
            'mission_w5_update',
            '미션 1 달성: 순전파와 역전파가 반복되며 전체 오차가 줄어들고 있습니다.'
          );
        }
        renderAll();
        if (local.appliedUpdates >= 60) {
          stopLiveTraining();
        }
      }
    }, 340);
  }

  function syncSlidersFromLocal() {
    slI1.value = String(local.inputs[0]);
    slI2.value = String(local.inputs[1]);
    slT1.value = String(local.targets[0]);
    slLr.value = String(local.lr);
    lblI1.textContent = local.inputs[0].toFixed(2);
    lblI2.textContent = local.inputs[1].toFixed(2);
    lblT1.textContent = local.targets[0].toFixed(2);
    lblLr.textContent = local.lr.toFixed(2);
  }

  // ===== 5. 스파크라인 & 전체 렌더링 =====
  function renderSparkline() {
    const hist = local.errorHistory;
    const latest = hist[hist.length - 1];
    sparkValEl.textContent = fmt(latest, 6);

    const w = 200;
    const h = 34;
    const pad = 4;
    const maxV = Math.max(...hist, 0.25);
    const minV = Math.min(...hist, 0);
    const span = Math.max(0.0005, maxV - minV);

    const pts = hist.map((v, i) => {
      const x = hist.length === 1 ? w - pad : pad + (i / (hist.length - 1)) * (w - pad * 2);
      const y = h - pad - ((v - minV) / span) * (h - pad * 2);
      return [x, y];
    });

    const linePath = pts.map(([x, y], i) => `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`).join(' ');
    const areaPath =
      pts.length > 1
        ? `${linePath} L ${pts[pts.length - 1][0].toFixed(1)} ${h} L ${pts[0][0].toFixed(1)} ${h} Z`
        : '';
    const last = pts[pts.length - 1];

    sparkSvgEl.innerHTML = `
      <defs>
        <linearGradient id="spark-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#10b981" stop-opacity="0.35" />
          <stop offset="100%" stop-color="#10b981" stop-opacity="0.0" />
        </linearGradient>
      </defs>
      ${areaPath ? `<path d="${areaPath}" fill="url(#spark-grad)" />` : ''}
      ${pts.length > 1 ? `<path d="${linePath}" fill="none" stroke="#10b981" stroke-width="2.2" stroke-linecap="round" />` : ''}
      <circle cx="${last[0].toFixed(1)}" cy="${last[1].toFixed(1)}" r="3.5" fill="#10b981" />
    `;
  }

  function renderAll() {
    const sim = getSim(false);
    const { before, after, selected } = sim;

    neuralCanvas.setState({
      sim,
      W1: local.W1,
      W2: local.W2,
      stage: local.stage,
      weightKey: local.selectedWeight,
      phase: local.flowPhase,
      activeInspectId: local.inspectedId,
    });
    renderInspector();
    renderKeywordBar();

    // 슬라이드 트랙 위치 이동 & 현재 슬라이드 높이에 맞춰 뷰포트 자동 축소/확장 (불필요한 여백 제거)
    slideTrackEl.style.transform = `translateX(-${local.slideIdx * 100}%)`;
    requestAnimationFrame(syncSlideViewportHeight);
    slideIndicatorEl.textContent = `${local.slideIdx + 1} / ${SLIDE_META.length}`;
    prevSlideBtn.disabled = local.slideIdx === 0;
    nextSlideBtn.disabled = local.slideIdx === SLIDE_META.length - 1;

    slideTabsEl.querySelectorAll('.slide-tab-btn').forEach((btn) => {
      const idx = Number(btn.dataset.slideTab);
      btn.classList.toggle('is-active', idx === local.slideIdx);
    });

    const isBackNow = local.flowPhase === 'backward' || (local.flowPhase === 'auto' && local.stage >= 3);
    modeFwdBtn.classList.toggle('is-active-fwd', !isBackNow);
    modeBackBtn.classList.toggle('is-active-back', isBackNow);

    rootEl.querySelectorAll('[data-weight]').forEach((btn) => {
      btn.classList.toggle('is-active', btn.dataset.weight === local.selectedWeight);
    });
    stagePillsEl.querySelectorAll('.stage-pill-btn').forEach((btn) => {
      const s = Number(btn.dataset.stagePill);
      btn.classList.toggle('is-active', s === local.stage);
      btn.classList.toggle('is-back-pill', s >= 3);
      btn.classList.toggle('is-passed', s < local.stage);
    });

    presetBarEl.hidden = !local.presentationMode;
    updateCounterEl.textContent = `${local.appliedUpdates}회`;

    // 3단계 이퀄라이저 카드 & 실시간 게이지 바
    const signHint =
      selected.step1 > 0.0005
        ? '출력 > 목표 (줄임 ↓)'
        : selected.step1 < -0.0005
          ? '출력 < 목표 (키움 ↑)'
          : '목표와 일치';

    valS1.textContent = fmt(selected.step1, 6);
    valS2.textContent = fmt(selected.step2, 6);
    valS3.textContent = fmt(selected.step3, 6);
    valGrad.textContent = fmt(selected.gradient, 6);

    barS1.style.width = `${Math.min(100, Math.abs(selected.step1) * 100).toFixed(1)}%`;
    barS2.style.width = `${Math.min(100, (selected.step2 / 0.25) * 100).toFixed(1)}%`;
    barS3.style.width = `${Math.min(100, selected.step3 * 100).toFixed(1)}%`;
    barGrad.style.width = `${Math.min(100, Math.abs(selected.gradient) * 650).toFixed(1)}%`;

    descS1.textContent = signHint;
    descS2.textContent = `최대 0.25의 ${((selected.step2 / 0.25) * 100).toFixed(0)}%`;
    descS3.textContent = `${selected.meta.from} 출력 신호`;
    descGrad.textContent = `${selected.meta.key} 최종 기울기`;

    rootEl.querySelectorAll('.chain-card').forEach((card) => {
      const cs = Number(card.dataset.chainStep);
      card.classList.toggle('is-current', local.stage === cs);
      card.classList.toggle('is-revealed', local.stage >= cs);
    });

    // 수정 미리보기 HUD
    const dirWord =
      selected.deltaWeight < -1e-7 ? '감소 ↓' : selected.deltaWeight > 1e-7 ? '증가 ↑' : '유지';
    const dirClass =
      selected.deltaWeight < -1e-7 ? 'is-down' : selected.deltaWeight > 1e-7 ? 'is-up' : '';

    weightChangeLabel.textContent = `${selected.meta.label} 수정 미리보기 (lr ${local.lr.toFixed(2)})`;
    weightChangeRow.innerHTML = `
      <span class="num">${fmt(selected.oldWeight, 6)}</span>
      <span class="arrow-inline">→</span>
      <strong class="num text-fwd">${fmt(selected.newWeight, 6)}</strong>
      <span class="delta-badge ${dirClass}">${selected.deltaWeight >= 0 ? '+' : ''}${fmt(selected.deltaWeight, 6)} (${dirWord})</span>
    `;
    applyOneBtn.textContent = `⚡ ${selected.meta.key} 1회 수정 발사`;

    // 슬라이드 내 실시간 도식
    rootEl.querySelector('#m5-ch1-diagram').innerHTML = renderChapter1Diagram(before);
    rootEl.querySelector('#m5-ch2-diagram').innerHTML = renderChapter2Diagram(before);
    rootEl.querySelector('#m5-ch3-diagram').innerHTML = renderChapter3Diagram(selected);
    rootEl.querySelector('#m5-ch4-diagram').innerHTML = renderChapter4Diagram(before, selected);
    rootEl.querySelector('#m5-ch5-diagram').innerHTML = renderChapter5Diagram(selected);

    // 오차 비교 박스
    const diffErr = before.eTotal - after.eTotal;
    const maxE = Math.max(before.eTotal, after.eTotal, 0.001);
    errorCompareEl.innerHTML = `
      <div class="err-bar-line">
        <span>수정 전</span>
        <div class="err-track"><div class="err-fill is-before" style="width:${((before.eTotal / maxE) * 100).toFixed(1)}%"></div></div>
        <span class="num">${fmt(before.eTotal, 6)}</span>
      </div>
      <div class="err-bar-line">
        <span>수정 후</span>
        <div class="err-track"><div class="err-fill is-after" style="width:${((after.eTotal / maxE) * 100).toFixed(1)}%"></div></div>
        <strong class="num">${fmt(after.eTotal, 6)}</strong>
      </div>
      <div class="err-diff-note">${
        diffErr > 0
          ? `1회 수정 시 전체 오차가 ${fmt(diffErr, 6)} 줄어듭니다.`
          : diffErr < 0
            ? `학습률이 커서 오차가 ${fmt(-diffErr, 6)} 커집니다.`
            : '오차 변화가 없습니다.'
      }</div>
    `;

    // 접이식 수식
    const oName = selected.meta.to;
    const hName = selected.meta.from;
    const wName = selected.meta.key;
    formulaBodyEl.innerHTML = `
      <p class="formula-line">
        ∂E<sub>total</sub> / ∂${wName}
        = (∂E<sub>total</sub> / ∂out<sub>${oName}</sub>)
        × (∂out<sub>${oName}</sub> / ∂net<sub>${oName}</sub>)
        × (∂net<sub>${oName}</sub> / ∂${wName})
      </p>
      <p class="formula-line num">
        = ${fmt(selected.step1, 6)} × ${fmt(selected.step2, 6)} × ${fmt(selected.step3, 6)}
        = <strong>${fmt(selected.gradient, 6)}</strong>
      </p>
      <p class="formula-line num">
        ${wName}(new) = ${fmt(selected.oldWeight, 6)} − ${local.lr.toFixed(2)} × (${fmt(selected.gradient, 6)})
        = <strong>${fmt(selected.newWeight, 6)}</strong>
      </p>
    `;

    renderSparkline();
  }

  function renderMissionsAndCompare() {
    const mod = store.getModuleState('m5');
    const missionListEl = rootEl.querySelector('#m5-mission-list');
    missionListEl.innerHTML = M5_CONTENT.missions
      .map((m) => {
        const done = Boolean(mod.missions[m.id]);
        return `
          <div class="mission-item ${done ? 'is-done' : ''}">
            <span class="mission-icon">${done ? '✓' : '○'}</span>
            <div>
              <div class="mission-title">${m.title}</div>
              <div class="mission-desc">${m.desc}</div>
            </div>
          </div>
        `;
      })
      .join('');

    const predCompareEl = rootEl.querySelector('#m5-pred-compare');
    if (mod.prediction?.choiceId) {
      const matched = mod.prediction.choiceId === M5_CONTENT.prediction.expectedId;
      predCompareEl.hidden = false;
      predCompareEl.innerHTML = `
        <div class="compare-title">처음 예상과 비교</div>
        <p>처음 고른 답: <strong>"${mod.prediction.choiceLabel}"</strong> (${
          matched ? '실제 결과와 일치합니다.' : '실험 결과와 달랐습니다.'
        })</p>
        <p class="compare-note">${M5_CONTENT.prediction.comparisonNote}</p>
      `;
    } else {
      predCompareEl.hidden = true;
    }
  }

  // ===== 6. 예측 · 확인 체크 · 생각 질문 마운트 =====
  renderPredictionWidget(
    rootEl.querySelector('#m5-prediction-mount'),
    M5_CONTENT.prediction,
    savedMod.prediction,
    (choice) => {
      store.updateModule('m5', (mod) => {
        mod.prediction = choice;
      });
      renderMissionsAndCompare();
      showToast('예상이 저장되었습니다! 2번 슬라이드(#순전파)로 이동합니다.', 'info');
      setTimeout(() => goToSlide(1), 420);
    }
  );

  renderMcqWidget(
    rootEl.querySelector('#m5-check-1'),
    M5_CONTENT.checks[0],
    savedMod.checks[M5_CONTENT.checks[0].id],
    (ans) => {
      store.updateModule('m5', (mod) => {
        mod.checks[M5_CONTENT.checks[0].id] = ans;
      });
      renderMissionsAndCompare();
    }
  );

  renderKeywordWidget(
    rootEl.querySelector('#m5-check-2'),
    M5_CONTENT.checks[1],
    savedMod.checks[M5_CONTENT.checks[1].id],
    (ans) => {
      store.updateModule('m5', (mod) => {
        mod.checks[M5_CONTENT.checks[1].id] = ans;
      });
      renderMissionsAndCompare();
    }
  );

  const reflectionInput = rootEl.querySelector('#m5-reflection-input');
  reflectionInput.value = savedMod.reflection || '';
  reflectionInput.addEventListener('input', () => {
    store.updateModule('m5', (mod) => {
      mod.reflection = reflectionInput.value;
    });
  });

  // ===== 7. 이벤트 바인딩 =====
  rootEl.querySelectorAll('[data-weight]').forEach((btn) => {
    btn.addEventListener('click', () => {
      selectWeight(btn.dataset.weight);
      const meta = getSim().selected.meta;
      neuralCanvas.triggerRipple(meta.to, '#f97316');
      neuralCanvas.triggerRipple(meta.from, '#3b82f6');
    });
  });

  applyOneBtn.addEventListener('click', () => {
    stopLiveTraining();
    const sim = getSim(false);
    local.W2 = sim.nextW2;
    local.appliedUpdates += 1;
    local.stage = 6;
    local.flowPhase = 'backward';
    local.masteredKeywords.add('kw_chain');
    local.errorHistory.push(sim.after.eTotal);
    if (local.errorHistory.length > 50) local.errorHistory.shift();

    neuralCanvas.triggerWeightUpdateEffect(
      local.selectedWeight,
      sim.selected.deltaWeight,
      sim.errorReduction
    );

    recordInteraction((mod) => {
      mod.interactions.updateAppliedCount += 1;
    });
    if (local.selectedWeight === 'w5' && sim.errorReduction > 0) {
      completeMission(
        'mission_w5_update',
        `미션 1 달성: w5 수정으로 전체 오차가 ${fmt(sim.errorReduction, 6)} 줄었습니다.`
      );
    } else if (local.selectedWeight === 'w6') {
      completeMission('mission_w6_sign', '미션 2 달성: w6 값이 커지면서 오차가 줄었습니다.');
    }
    renderAll();
    flashElement(updateCounterEl);
  });

  rootEl.querySelector('#m5-apply-all').addEventListener('click', () => {
    stopLiveTraining();
    const sim = getSim(true);
    local.W2 = sim.nextW2;
    local.appliedUpdates += 1;
    local.stage = 6;
    local.flowPhase = 'backward';
    local.masteredKeywords.add('kw_chain');
    local.errorHistory.push(sim.after.eTotal);
    if (local.errorHistory.length > 50) local.errorHistory.shift();

    for (const k of ['w5', 'w6', 'w7', 'w8']) {
      neuralCanvas.triggerWeightUpdateEffect(k, sim.allDetails[k].deltaWeight, 0);
    }
    neuralCanvas.spawnFloatingText(1367, 455, `전체 오차 −${fmt(sim.errorReduction, 5)} 감소!`, '#10b981');

    recordInteraction((mod) => {
      mod.interactions.allWeightsAppliedCount += 1;
    });
    if (sim.errorReduction > 0) {
      completeMission(
        'mission_w5_update',
        `미션 1 달성: w5~w8 일괄 수정으로 오차가 ${fmt(sim.before.eTotal, 6)} → ${fmt(sim.after.eTotal, 6)}로 줄었습니다.`
      );
    }
    renderAll();
    flashElement(updateCounterEl);
  });

  liveLoopBtn.addEventListener('click', toggleLiveTraining);

  rootEl.querySelector('#m5-reset-weights').addEventListener('click', () => {
    stopLiveTraining();
    local.W1 = cloneMatrix(DEFAULT_PARAMS.W1);
    local.W2 = cloneMatrix(DEFAULT_PARAMS.W2);
    local.appliedUpdates = 0;
    local.errorHistory = [getSim().before.eTotal];
    renderAll();
    showToast('가중치를 초기값으로 되돌렸습니다.', 'info');
  });

  function onSliderChange() {
    local.inputs = [Number(slI1.value), Number(slI2.value)];
    local.targets = [Number(slT1.value), DEFAULT_PARAMS.targets[1]];
    local.lr = Number(slLr.value);
    lblI1.textContent = local.inputs[0].toFixed(2);
    lblI2.textContent = local.inputs[1].toFixed(2);
    lblT1.textContent = local.targets[0].toFixed(2);
    lblLr.textContent = local.lr.toFixed(2);
    local.errorHistory[local.errorHistory.length - 1] = getSim().before.eTotal;
    recordInteraction((mod) => {
      mod.interactions.sliderChanges += 1;
    });
    renderAll();
  }
  [slI1, slI2, slT1, slLr].forEach((sl) => sl.addEventListener('input', onSliderChange));

  rootEl.querySelector('#m5-reset-all').addEventListener('click', () => {
    stopLiveTraining();
    local.inputs = [...DEFAULT_PARAMS.inputs];
    local.targets = [...DEFAULT_PARAMS.targets];
    local.W1 = cloneMatrix(DEFAULT_PARAMS.W1);
    local.W2 = cloneMatrix(DEFAULT_PARAMS.W2);
    local.lr = DEFAULT_PARAMS.lr;
    local.selectedWeight = 'w5';
    local.appliedUpdates = 0;
    local.flowPhase = 'auto';
    local.errorHistory = [getSim().before.eTotal];
    syncSlidersFromLocal();
    renderAll();
    showToast('모든 값을 기본값으로 복원했습니다.', 'info');
  });

  rootEl.querySelector('#m5-panel-submit-btn').addEventListener('click', () => {
    if (onOpenSubmit) onOpenSubmit();
  });

  syncSlidersFromLocal();
  renderAll();
  renderMissionsAndCompare();
  syncSlideViewportHeight();

  return {
    setPresentationMode(enabled) {
      local.presentationMode = Boolean(enabled);
      renderAll();
    },
    destroy() {
      stopLiveTraining();
      if (slideResizeObserver) slideResizeObserver.disconnect();
      destroyEntropy();
      neuralCanvas.destroy();
      window.removeEventListener('keydown', onGlobalKeyDown);
    },
  };
}
