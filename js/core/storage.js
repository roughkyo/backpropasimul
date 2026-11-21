/**
 * 학습 기록 로컬 자동 저장 + Supabase REST/RPC 연동 클라이언트 (외부 SDK 없음)
 */
import { APP_CONFIG } from '../../config.js';

const STORAGE_KEY_PREFIX = 'sim_learning_state_v1::';
const DEMO_SUBMISSIONS_KEY = 'sim_demo_submissions_v1::';

const memoryStore = new Map();

function safeGetItem(key) {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return memoryStore.get(key) ?? null;
  }
}

function safeSetItem(key, value) {
  try {
    window.localStorage.setItem(key, value);
    return true;
  } catch {
    memoryStore.set(key, value);
    return false;
  }
}

export function isSupabaseConfigured() {
  return Boolean(
    APP_CONFIG.supabaseUrl &&
      APP_CONFIG.supabaseUrl.trim().startsWith('http') &&
      APP_CONFIG.supabaseAnonKey &&
      APP_CONFIG.supabaseAnonKey.trim().length > 10
  );
}

function createInitialState() {
  return {
    lessonId: APP_CONFIG.lessonId,
    startedAt: new Date().toISOString(),
    lastActiveAt: new Date().toISOString(),
    durationSec: 0,
    studentProfile: {
      studentId: '',
      studentName: '',
    },
    lastSubmission: null,
    modules: {
      m5: {
        prediction: null,
        missions: {},
        checks: {},
        reflection: '',
        interactions: {
          stepVisited: [0],
          weightsInspected: ['w5'],
          updateAppliedCount: 0,
          allWeightsAppliedCount: 0,
          sliderChanges: 0,
        },
        completed: false,
      },
    },
  };
}

export class LearningStore {
  constructor() {
    this.key = `${STORAGE_KEY_PREFIX}${APP_CONFIG.lessonId}`;
    this.state = this.load();
    this.listeners = new Set();
    this.sessionStartedMs = Date.now();
  }

  load() {
    const raw = safeGetItem(this.key);
    const base = createInitialState();
    if (!raw) return base;
    try {
      const parsed = JSON.parse(raw);
      return {
        ...base,
        ...parsed,
        studentProfile: { ...base.studentProfile, ...(parsed.studentProfile || {}) },
        modules: {
          ...base.modules,
          ...(parsed.modules || {}),
          m5: {
            ...base.modules.m5,
            ...(parsed.modules?.m5 || {}),
            missions: { ...(parsed.modules?.m5?.missions || {}) },
            checks: { ...(parsed.modules?.m5?.checks || {}) },
            interactions: {
              ...base.modules.m5.interactions,
              ...(parsed.modules?.m5?.interactions || {}),
            },
          },
        },
      };
    } catch {
      return base;
    }
  }

  save() {
    this.state.lastActiveAt = new Date().toISOString();
    safeSetItem(this.key, JSON.stringify(this.state));
    for (const fn of this.listeners) {
      fn(this.state);
    }
  }

  subscribe(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  getModuleState(moduleId = 'm5') {
    if (!this.state.modules[moduleId]) {
      this.state.modules[moduleId] = createInitialState().modules.m5;
    }
    return this.state.modules[moduleId];
  }

  updateModule(moduleId, updater) {
    const mod = this.getModuleState(moduleId);
    updater(mod);
    mod.completed = Boolean(
      mod.prediction?.choiceId &&
        Object.values(mod.missions).some(Boolean) &&
        Object.values(mod.checks).filter((c) => c?.correct).length >= 2
    );
    this.save();
  }

  buildPayload() {
    const elapsedSec = Math.max(
      this.state.durationSec || 0,
      Math.round((Date.now() - this.sessionStartedMs) / 1000)
    );
    this.state.durationSec = elapsedSec;
    return {
      lessonId: APP_CONFIG.lessonId,
      lessonTitle: APP_CONFIG.lessonTitle,
      startedAt: this.state.startedAt,
      submittedClientAt: new Date().toISOString(),
      durationSec: elapsedSec,
      modules: this.state.modules,
    };
  }

  async submit({ studentId, studentName, password }) {
    const cleanId = studentId.trim();
    const cleanName = studentName.trim();
    const cleanPw = password.trim();

    if (!cleanId) return { ok: false, message: '학번을 입력해 주세요.' };
    if (!cleanName) return { ok: false, message: '이름을 입력해 주세요.' };
    if (cleanPw.length < 4) return { ok: false, message: '비밀번호는 4자 이상 입력해 주세요.' };

    this.state.studentProfile = { studentId: cleanId, studentName: cleanName };
    const payload = this.buildPayload();

    // Supabase가 아직 설정되지 않은 경우 로컬 데모 제출로 안전 처리
    if (!isSupabaseConfigured()) {
      const demoKey = `${DEMO_SUBMISSIONS_KEY}${APP_CONFIG.lessonId}`;
      let list = [];
      try {
        list = JSON.parse(safeGetItem(demoKey) || '[]');
      } catch {
        list = [];
      }
      const existingIdx = list.findIndex((item) => item.student_id === cleanId);
      const pwToken = btoa(unescape(encodeURIComponent(cleanPw)));
      const nowIso = new Date().toISOString();

      if (existingIdx >= 0) {
        if (list[existingIdx].demo_pw_token !== pwToken) {
          return {
            ok: false,
            error: 'INVALID_PASSWORD',
            message: '이미 제출된 학번입니다. 처음 제출할 때 입력한 비밀번호와 일치하지 않습니다.',
          };
        }
        list[existingIdx] = {
          ...list[existingIdx],
          student_name: cleanName,
          payload,
          submit_count: (list[existingIdx].submit_count || 1) + 1,
          updated_at: nowIso,
        };
      } else {
        list.push({
          lesson_id: APP_CONFIG.lessonId,
          student_id: cleanId,
          student_name: cleanName,
          demo_pw_token: pwToken,
          payload,
          submit_count: 1,
          submitted_at: nowIso,
          updated_at: nowIso,
        });
      }
      safeSetItem(demoKey, JSON.stringify(list));

      const record = existingIdx >= 0 ? list[existingIdx] : list[list.length - 1];
      this.state.lastSubmission = {
        mode: 'local_demo',
        studentId: cleanId,
        studentName: cleanName,
        submitCount: record.submit_count,
        updatedAt: nowIso,
      };
      this.save();

      return {
        ok: true,
        mode: 'local_demo',
        submit_count: record.submit_count,
        updated_at: nowIso,
        message:
          '로컬 데모 모드로 저장되었습니다. (config.js에 Supabase URL과 anon key를 입력하면 실제 교사 DB로 전송됩니다.)',
      };
    }

    // 실제 Supabase RPC 호출 (외부 SDK 없이 브라우저 내장 fetch 사용)
    const baseUrl = APP_CONFIG.supabaseUrl.replace(/\/+$/, '');
    const key = APP_CONFIG.supabaseAnonKey.trim();
    const headers = {
      'Content-Type': 'application/json',
      apikey: key,
    };
    if (!key.startsWith('sb_')) {
      headers.Authorization = `Bearer ${key}`;
    }
    try {
      const res = await fetch(`${baseUrl}/rest/v1/rpc/submit_learning_log`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          p_lesson_id: APP_CONFIG.lessonId,
          p_student_id: cleanId,
          p_student_name: cleanName,
          p_password: cleanPw,
          p_payload: payload,
        }),
      });

      const data = await res.json().catch(() => null);
      if (!res.ok) {
        return {
          ok: false,
          message:
            data?.message ||
            data?.hint ||
            `서버 응답 오류(${res.status})가 발생했습니다. 작업 내용은 브라우저에 안전하게 남아 있으니 잠시 후 다시 제출해 주세요.`,
        };
      }

      if (!data?.ok) {
        return {
          ok: false,
          error: data?.error || 'SUBMIT_REJECTED',
          message: data?.message || '제출이 거부되었습니다. 입력 정보를 확인해 주세요.',
        };
      }

      this.state.lastSubmission = {
        mode: data.mode || 'submitted',
        studentId: cleanId,
        studentName: cleanName,
        submitCount: data.submit_count || 1,
        updatedAt: data.updated_at || new Date().toISOString(),
      };
      this.save();

      return data;
    } catch {
      return {
        ok: false,
        error: 'NETWORK_ERROR',
        message:
          '네트워크 연결을 확인해 주세요. 작업 기록은 이 브라우저에 안전하게 보관되어 있으므로 연결 후 [다시 제출]할 수 있습니다.',
      };
    }
  }
}

/**
 * 교사용 페이지(teacher.html) Supabase Auth 로그인 및 제출 목록 조회
 */
export async function teacherSignIn(email, password) {
  if (!isSupabaseConfigured()) {
    throw new Error('config.js에 Supabase URL과 anon key가 설정되지 않았습니다.');
  }
  const baseUrl = APP_CONFIG.supabaseUrl.replace(/\/+$/, '');
  const res = await fetch(`${baseUrl}/auth/v1/token?grant_type=password`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      apikey: APP_CONFIG.supabaseAnonKey,
    },
    body: JSON.stringify({ email: email.trim(), password }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || !data.access_token) {
    throw new Error(data.error_description || data.msg || '교사 계정 로그인에 실패했습니다.');
  }
  return data;
}

export async function fetchTeacherSubmissions(accessToken, lessonId = '') {
  if (!isSupabaseConfigured()) {
    // 로컬 데모 데이터 조회
    const demoKey = `${DEMO_SUBMISSIONS_KEY}${lessonId || APP_CONFIG.lessonId}`;
    try {
      return JSON.parse(safeGetItem(demoKey) || '[]');
    } catch {
      return [];
    }
  }
  const baseUrl = APP_CONFIG.supabaseUrl.replace(/\/+$/, '');
  const query = new URLSearchParams({
    select: 'id,lesson_id,student_id,student_name,submit_count,submitted_at,updated_at,payload',
    order: 'updated_at.desc',
  });
  if (lessonId) {
    query.set('lesson_id', `eq.${lessonId}`);
  }
  const res = await fetch(`${baseUrl}/rest/v1/teacher_submission_summary?${query.toString()}`, {
    headers: {
      apikey: APP_CONFIG.supabaseAnonKey,
      Authorization: `Bearer ${accessToken}`,
    },
  });
  const data = await res.json().catch(() => []);
  if (!res.ok) {
    throw new Error(data?.message || `조회 실패 (${res.status})`);
  }
  return Array.isArray(data) ? data : [];
}
