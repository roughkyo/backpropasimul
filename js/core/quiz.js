/**
 * 공통 체크 엔진 — 예측형 객관식, 확인형 객관식, 키워드형, 생각 질문
 */

export function normalizeText(str) {
  return String(str || '')
    .toLowerCase()
    .replace(/\s+/g, '')
    .replace(/[.,!?'"()\[\]{}\-_/]/g, '');
}

/**
 * 키워드형 채점: 띄어쓰기·대소문자 무시, 동의어 목록 매칭, 부분 포함 안내
 */
export function evaluateKeywordAnswer(rawAnswer, keywordGroups) {
  const norm = normalizeText(rawAnswer);
  if (!norm) {
    return {
      empty: true,
      correct: false,
      matchedCount: 0,
      totalCount: keywordGroups.length,
      matchedCanonicals: [],
    };
  }

  const matchedCanonicals = [];
  for (const group of keywordGroups) {
    const candidates = [group.canonical, ...(group.synonyms || [])].map(normalizeText);
    const hit = candidates.some((cand) => cand && norm.includes(cand));
    if (hit) {
      matchedCanonicals.push(group.canonical);
    }
  }

  return {
    empty: false,
    correct: matchedCanonicals.length === keywordGroups.length,
    matchedCount: matchedCanonicals.length,
    totalCount: keywordGroups.length,
    matchedCanonicals,
  };
}

/**
 * 예측형 객관식 렌더링 (조작 전, 정답 비공개)
 */
export function renderPredictionWidget(container, prediction, savedPrediction, onSelect) {
  container.innerHTML = '';

  const qEl = document.createElement('p');
  qEl.className = 'quiz-question';
  qEl.textContent = prediction.question;
  container.appendChild(qEl);

  const list = document.createElement('div');
  list.className = 'quiz-options';
  list.setAttribute('role', 'radiogroup');
  list.setAttribute('aria-label', '예측 선택지');

  prediction.options.forEach((opt, idx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'quiz-option-btn';
    if (savedPrediction?.choiceId === opt.id) {
      btn.classList.add('is-selected');
    }
    btn.innerHTML = `<span class="opt-index">${idx + 1}</span><span>${opt.label}</span>`;
    btn.addEventListener('click', () => {
      list.querySelectorAll('.quiz-option-btn').forEach((b) => b.classList.remove('is-selected'));
      btn.classList.add('is-selected');
      onSelect({
        choiceId: opt.id,
        choiceLabel: opt.label,
        selectedAt: new Date().toISOString(),
      });
    });
    list.appendChild(btn);
  });

  container.appendChild(list);

  const note = document.createElement('p');
  note.className = 'quiz-helper';
  note.textContent = savedPrediction?.choiceId
    ? `선택한 예상: "${savedPrediction.choiceLabel}" — 아래 시뮬레이션으로 실제 결과를 확인해 보세요.`
    : '맞히지 않아도 괜찮습니다. 먼저 결과를 짐작한 뒤 아래 실험과 비교해 봅니다.';
  container.appendChild(note);
}

/**
 * 확인형 객관식 렌더링 (오답 보기별 맞춤형 오개념 피드백, 무제한 재시도)
 */
export function renderMcqWidget(container, check, savedCheck, onAnswer) {
  container.innerHTML = '';

  const qEl = document.createElement('p');
  qEl.className = 'quiz-question';
  qEl.textContent = check.question;
  container.appendChild(qEl);

  const list = document.createElement('div');
  list.className = 'quiz-options';

  const feedbackBox = document.createElement('div');
  feedbackBox.className = 'quiz-feedback';
  feedbackBox.setAttribute('aria-live', 'polite');

  const applyFeedback = (opt) => {
    feedbackBox.hidden = false;
    feedbackBox.className = `quiz-feedback ${opt.correct ? 'is-correct' : 'is-wrong'}`;
    feedbackBox.innerHTML = `<strong>${opt.correct ? '정답' : '다시 확인'}</strong> ${opt.feedback}`;
  };

  check.options.forEach((opt, idx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'quiz-option-btn';
    if (savedCheck?.selectedId === opt.id) {
      btn.classList.add(opt.correct ? 'is-correct' : 'is-wrong');
    }
    btn.innerHTML = `<span class="opt-index">${idx + 1}</span><span>${opt.label}</span>`;
    btn.addEventListener('click', () => {
      list.querySelectorAll('.quiz-option-btn').forEach((b) => {
        b.classList.remove('is-correct', 'is-wrong');
      });
      btn.classList.add(opt.correct ? 'is-correct' : 'is-wrong');
      applyFeedback(opt);
      onAnswer({
        type: 'mcq',
        selectedId: opt.id,
        selectedLabel: opt.label,
        correct: Boolean(opt.correct),
        attempts: (savedCheck?.attempts || 0) + 1,
        answeredAt: new Date().toISOString(),
      });
    });
    list.appendChild(btn);
  });

  container.appendChild(list);

  if (savedCheck?.selectedId) {
    const prevOpt = check.options.find((o) => o.id === savedCheck.selectedId);
    if (prevOpt) applyFeedback(prevOpt);
  } else {
    feedbackBox.hidden = true;
  }
  container.appendChild(feedbackBox);
}

/**
 * 키워드형 체크 렌더링
 */
export function renderKeywordWidget(container, check, savedCheck, onAnswer) {
  container.innerHTML = '';

  const qEl = document.createElement('p');
  qEl.className = 'quiz-question';
  qEl.textContent = check.question;
  container.appendChild(qEl);

  const row = document.createElement('form');
  row.className = 'keyword-row';

  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'text-input';
  input.placeholder = check.placeholder || '답을 입력하세요';
  input.value = savedCheck?.answerText || '';
  input.setAttribute('aria-label', check.question);

  const submitBtn = document.createElement('button');
  submitBtn.type = 'submit';
  submitBtn.className = 'btn btn-secondary';
  submitBtn.textContent = '확인';

  row.appendChild(input);
  row.appendChild(submitBtn);
  container.appendChild(row);

  const feedbackBox = document.createElement('div');
  feedbackBox.className = 'quiz-feedback';
  feedbackBox.setAttribute('aria-live', 'polite');

  const showResult = (result) => {
    feedbackBox.hidden = false;
    if (result.empty) {
      feedbackBox.className = 'quiz-feedback is-wrong';
      feedbackBox.textContent = '단어나 구를 입력한 뒤 확인을 눌러 주세요.';
      return;
    }
    if (result.correct) {
      feedbackBox.className = 'quiz-feedback is-correct';
      feedbackBox.innerHTML = `<strong>정답</strong> ${check.successFeedback}`;
    } else if (result.matchedCount > 0) {
      feedbackBox.className = 'quiz-feedback is-wrong';
      feedbackBox.innerHTML = `<strong>일부 포함</strong> 핵심어 ${result.totalCount}개 중 ${result.matchedCount}개가 들어 있습니다. ${check.failFeedback}`;
    } else {
      feedbackBox.className = 'quiz-feedback is-wrong';
      feedbackBox.innerHTML = `<strong>다시 확인</strong> ${check.failFeedback}`;
    }
  };

  if (savedCheck?.answerText) {
    const prevRes = evaluateKeywordAnswer(savedCheck.answerText, check.keywords);
    showResult(prevRes);
  } else {
    feedbackBox.hidden = true;
  }

  let attemptCount = savedCheck?.attempts || 0;
  row.addEventListener('submit', (e) => {
    e.preventDefault();
    const res = evaluateKeywordAnswer(input.value, check.keywords);
    showResult(res);
    if (!res.empty) {
      attemptCount += 1;
      onAnswer({
        type: 'keyword',
        answerText: input.value.trim(),
        correct: res.correct,
        attempts: attemptCount,
        answeredAt: new Date().toISOString(),
      });
    }
  });

  container.appendChild(feedbackBox);
}
