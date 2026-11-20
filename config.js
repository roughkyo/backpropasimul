/**
 * 수업별 설정 파일 (범용 재사용 포인트)
 *
 * 1) lessonId: 수업자료/차시별로 고유 ID를 부여하면 하나의 Supabase 프로젝트로
 *    여러 차시의 학생 제출 기록을 통합 관리할 수 있습니다.
 * 2) supabaseUrl / supabaseAnonKey: Supabase 프로젝트 설정(Project Settings -> API)의
 *    Project URL과 anon(public) key를 입력하세요.
 *    비워 두면 오프라인/로컬 데모 모드로 동작하며, 학생 기록은 브라우저에 안전 보관됩니다.
 */
export const APP_CONFIG = {
  lessonId: 'dl-ch1-m5-backprop',
  lessonTitle: '1차시 딥러닝 — M5. 역전파 한 걸음씩',
  lessonSubtitle: '출력에서 생긴 오차가 어떻게 가중치 w5까지 거슬러 올라갈까?',
  supabaseUrl: '', // 예: 'https://xxxxxxxxxxxx.supabase.co'
  supabaseAnonKey: '', // 예: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...'
};
