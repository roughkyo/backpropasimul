/**
 * 수업별 설정 파일 (범용 재사용 포인트)
 *
 * 1) lessonId: 수업자료/차시별로 고유 ID를 부여하면 하나의 Supabase 프로젝트로
 *    여러 차시의 학생 제출 기록을 통합 관리할 수 있습니다.
 * 2) supabaseUrl / supabaseAnonKey: Supabase 프로젝트 연결 설정
 */
export const APP_CONFIG = {
  lessonId: 'dl-ch1-m5-backprop',
  lessonTitle: '1차시 딥러닝 — M5. 역전파 한 걸음씩',
  lessonSubtitle: '출력에서 생긴 오차가 어떻게 가중치 w5까지 거슬러 올라갈까?',
  supabaseUrl: 'https://bgnacpqniwamixrhouue.supabase.co',
  supabaseAnonKey: 'sb_publishable_87YLvz7StnyK1FrYYV0HJg_MSCDRwOj',
};
