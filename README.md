# 1차시 딥러닝 — M5. 역전파 한 걸음씩 (Interactive Backpropagation Simulation)

고등학교 1~2학년 인공지능·정보 수업을 위해 제작된 **인터랙티브 역전파(Backpropagation) 시뮬레이션 교수학습 웹페이지**입니다.  
설치나 회원가입 없이 브라우저에서 즉시 실행되며, **화이트 모드(기본, 교실 프로젝터 최적화)**와 **다크 모드**를 모두 지원합니다.

---

## 1. 주요 기능

- **초대형 풀와이드(`1600×540`, 96vw) 60fps 라이브 신경망 캔버스**
  - **순전파(파랑 →)**와 **역전파(← 주황)**의 색상·방향을 분리한 실시간 입자 흐름 애니메이션
  - 각 뉴런 노드의 출력 비율(0~100%)을 시각화하는 원형 아크 게이지 및 `⛶ 크게 보기(극장 모드)` 지원
  - **전 요소(17개: 노드 6 + 가중치 8 + 오차 3) 마우스 호버 경로 추적** 및 **클릭 시 상세 부연설명 인스펙터** (현재 수치가 대입된 단계별 계산식과 실시간 시그모이드 미니 그래프 제공)
  - **`🔍 돋보기 (Z)`**: 마우스 주변을 2.2배 광학 확대하고 내부 계산값(`net`, 민감도 `σ'`, 전달 신호량, 기울기)을 X-Ray처럼 투시
- **키워드 단어 중심 학습 & 좌우 슬라이드 캐러셀 (`1 / 8 ~ 8 / 8`)**
  - 상단 **6대 핵심 키워드 마스터리 바**(`#순전파`, `#제곱오차`, `#오차방향`, `#시그모이드민감도`, `#입력신호책임`, `#체인룰·학습률`) 클릭 시 해당 애니메이션 즉시 발사 및 슬라이드 이동
  - **`⚡ w5 1회 수정 발사` / `🔥 w5~w8 일괄 수정` / `🚀 왕복 연속 학습`**: 실행 시 역방향 레이저 충격파, 플로팅 변화량(`w5 -0.0036 ↓`), 3단계 이퀄라이저 게이지, 실시간 전체 오차 스파크라인이 역동적으로 반응
- **수치 정확성 보장 (`scripts/verify_math.mjs`)**
  - 교사용 실습 노트북(`[1차시]퍼셉트론,딥러닝v5.2(교사용).ipynb` 3장 셀 53~71) 실행 결과와 **소수점 6자리까지 100% 일치** 자동 검증

---

## 2. 왜 Supabase에 배포·연동하는가? (아키텍처 및 도입 이유)

본 프로젝트는 프론트엔드를 **GitHub Pages(정적 웹 호스팅)**로 배포하고, 학습 데이터 수집 백엔드로 **Supabase(PostgreSQL + RLS + RPC)**를 연동합니다. Supabase를 사용하는 구체적인 이유는 다음과 같습니다.

### ① 정적 호스팅(GitHub Pages)의 한계 극복과 무서버(Serverless) 운영
- GitHub Pages는 빠르고 무료로 시뮬레이션 화면을 배포할 수 있지만, 학생들의 실습 기록을 저장할 데이터베이스와 서버 로직을 실행할 수 없습니다.
- 별도의 백엔드 서버(Node.js, Python 등)를 구축·유지보수하지 않고도 **PostgreSQL 데이터베이스, 인증(Auth), RPC 함수를 즉시 제공하는 Supabase**를 결합하여 운영 비용과 관리 부담을 최소화했습니다.

### ② 무가입(Zero-Friction) 학생 제출과 교사 실시간 형성평가 수집
- 수업 중 **5~10분** 안에 개념을 체험하고 넘어가야 하는 교실 환경에서 학생 회원가입이나 이메일 인증은 수업 흐름을 방해합니다.
- 학생은 평소 브라우저(`localStorage`)에 자동 저장하며 실습하다가, 마지막에 **`학번 · 이름 · 비밀번호(4자 이상)`**만 입력해 즉시 제출합니다.
- 교사는 `teacher.html`(또는 Supabase 대시보드)을 통해 **학생별 예측 선택, 미션 달성 여부, 확인 퀴즈 정답/시도 횟수, 생각 질문 서술, 체류 시간**을 실시간으로 모아 보고 **UTF-8 CSV 파일**로 내려받아 형성평가 및 수업 피드백에 활용할 수 있습니다.

### ③ 정적 웹 환경에서의 개인정보 보호 및 타인 학번 도용 방지 (RLS + `pgcrypto` 해시)
- 정적 웹페이지에는 Supabase `anon key`가 브라우저 코드에 포함됩니다. 만약 일반 테이블에 `SELECT`나 `UPDATE` 권한을 열어 두면 다른 학생의 제출 내용을 열람하거나 타인의 학번으로 기록을 덮어쓸 위험이 있습니다.
- 이를 방지하기 위해 **`learning_submissions` 테이블에 RLS(Row Level Security)를 활성화하여 익명(`anon`) 사용자의 직접 `SELECT / INSERT / UPDATE / DELETE`를 전면 차단**했습니다.
- 학생 제출은 오직 `SECURITY DEFINER` 권한의 **PostgreSQL RPC 함수(`submit_learning_log`)**를 통해서만 수행되며, 비밀번호는 평문으로 저장하지 않고 DB 내부의 **`pgcrypto`(`crypt` + `gen_salt('bf')`) bcrypt 해시**로 암호화하여 저장·검증합니다. 같은 학번으로 재제출할 때는 처음 입력한 비밀번호와 일치해야만 갱신됩니다.

### ④ 외부 JS SDK 의존성 제로 (`fetch` 기반 초경량 통신)
- 무거운 외부 라이브러리를 설치하지 않고 브라우저 내장 `fetch` API만으로 Supabase REST/RPC 엔드포인트(`/rest/v1/rpc/submit_learning_log`)를 호출하므로, 번들 도구 없이 순수 정적 파일로 동작하며 학교 크롬북·저사양 PC에서도 빠르게 로드됩니다.

### ⑤ 단일 Supabase 프로젝트로 다차시·다과목 통합 관리 (`lesson_id` 확장성)
- 테이블이 `(lesson_id, student_id)` 복합 고유키와 `payload JSONB` 구조로 설계되어 있습니다.
- 이후 다른 차시(예: 퍼셉트론 결정경계 `M1`, 경사하강법 `M4`, 신경망 학습 실험실 `M6` 또는 다른 `.ipynb`/`.py` 수업자료)를 추가할 때 새 DB를 만들 필요 없이 `config.js`의 `lessonId`만 변경하면 **하나의 Supabase 프로젝트에서 모든 차시의 학생 제출 기록을 통합 관리**할 수 있습니다.

---

## 3. 프로젝트 구조

```text
.
├─ index.html                  # 학생용 메인 시뮬레이션 페이지 (기본: 화이트 모드)
├─ teacher.html                # 교사용 제출 기록 조회 · 검색 · UTF-8 CSV 다운로드 페이지
├─ config.js                   # 차시 식별자(lessonId), Supabase URL 및 anon key 설정
├─ 설계명세서.md               # 범용 인터랙티브 시뮬레이션 교수학습 페이지 설계명세서
├─ css/
│  └─ style.css                # 화이트/다크 글래스모피즘 & 뉴모피즘 디자인 토큰, 슬라이드 레이아웃
├─ js/
│  ├─ main.js                  # 헤더 상태, 화이트/다크 테마 스위치, 발표 모드, 제출 모달 연결
│  ├─ core/
│  │  ├─ storage.js            # localStorage 자동 저장 + fetch 기반 Supabase RPC/Auth 클라이언트
│  │  ├─ quiz.js               # 예측형·확인 객관식·키워드형·생각 질문 체크 엔진
│  │  └─ ui.js                 # 토스트, 값 변화 강조, 작업 기록 제출 모달
│  ├─ math/
│  │  └─ backprop.js           # 2-2-2 시그모이드 신경망 순전파·오차·역전파 3단계 순수 계산 엔진
│  ├─ visuals/
│  │  ├─ neural-canvas.js      # 1600×540 60fps 초대형 신경망 캔버스 (호버·클릭·2.2× 돋보기·줌)
│  │  ├─ inspector.js          # 17개 전 요소 클릭 시 열리는 상세 부연설명 인스펙터
│  │  ├─ chapter-diagrams.js   # 슬라이드별 실시간 반응형 도식 (시그모이드 접선·오차 벡터·합류)
│  │  └─ entropy-bg.js         # 히어로 인터랙티브 파티클 캔버스
│  └─ modules/
│     └─ m5-backprop.js        # M5 키워드 마스터리 바, 3단계 이퀄라이저, 8단 슬라이드 캐러셀
├─ content/
│  └─ m5-backprop.js           # 교수학습 문구, 미션, 확인 퀴즈 및 맞춤 피드백 데이터 (코드와 분리)
├─ supabase/
│  ├─ schema.sql               # Supabase SQL Editor 실행용 테이블·RLS·RPC·뷰 통합 스크립트
│  └─ migrations/
│     └─ 20261121000000_learning_submissions.sql  # Supabase CLI 마이그레이션 파일
└─ scripts/
   └─ verify_math.mjs          # 노트북 기준값 소수점 6자리 자동 검증 스크립트
```

---

## 4. 배포 및 실행 방법

### 로컬 실행 및 수치 검증
```bash
# 1. 수학 계산 엔진 자동 검증 (소수점 6자리 대조)
node scripts/verify_math.mjs

# 2. 정적 웹 서버 실행
python -m http.server 8765
# 브라우저에서 http://localhost:8765/index.html 접속
```

### Supabase 배포 (2가지 방법 중 택 1)

1. **Supabase 대시보드 SQL Editor 이용 (가장 간편)**
   - Supabase 프로젝트 생성 후 **SQL Editor**에 [`supabase/schema.sql`](./supabase/schema.sql) 전체 내용을 붙여넣고 **Run**을 클릭합니다.
2. **Supabase CLI 이용**
   ```bash
   npx supabase login
   npx supabase link --project-ref <YOUR_PROJECT_REF>
   npx supabase db push
   ```
3. **프론트엔드 연결 (`config.js`)**
   - Supabase 대시보드의 `Project Settings → API`에서 **Project URL**과 **`anon` public key**를 복사해 [`config.js`](./config.js)의 `supabaseUrl`, `supabaseAnonKey`에 입력합니다.
   - 교사용 조회 페이지([`teacher.html`](./teacher.html))를 사용할 경우 Supabase `Authentication → Users`에서 교사 이메일/비밀번호 계정을 추가합니다.
