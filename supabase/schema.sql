-- ============================================================================
-- 인터랙티브 시뮬레이션 교수학습 페이지 — 범용 Supabase 스키마 (v0.2)
-- 어떤 차시/수업자료(.ipynb, .py)에도 lesson_id만 바꿔 재사용할 수 있습니다.
-- Supabase SQL Editor에 전체 복사 후 Run 하세요.
-- ============================================================================

-- 1. 비밀번호 안전 해시용 확장 활성화
create extension if not exists pgcrypto;

-- 2. 범용 학생 제출 기록 테이블
create table if not exists public.learning_submissions (
  id uuid primary key default gen_random_uuid(),
  lesson_id text not null,
  student_id text not null,
  student_name text not null,
  password_hash text not null,
  payload jsonb not null default '{}'::jsonb,
  submit_count integer not null default 1,
  submitted_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint uq_lesson_student unique (lesson_id, student_id)
);

create index if not exists idx_learning_submissions_lesson_updated
  on public.learning_submissions (lesson_id, updated_at desc);

-- 3. RLS(Row Level Security) 전면 활성화
-- 익명(anon) 사용자의 직접 SELECT/INSERT/UPDATE/DELETE를 모두 차단하여
-- 정적 사이트의 anon key로 타 학생 기록을 훔쳐보거나 덮어쓸 수 없게 합니다.
alter table public.learning_submissions enable row level security;

-- 기존 정책 재생성 대비
drop policy if exists "Teachers (authenticated) can read submissions" on public.learning_submissions;

-- 교사용 조회 정책: Supabase Auth로 로그인한 교사(authenticated)만 전체 조회 가능
-- (필요 시: and auth.jwt() ->> 'email' in ('teacher@school.kr') 조건을 추가해 특정 교사 계정으로 제한하세요.)
create policy "Teachers (authenticated) can read submissions"
  on public.learning_submissions
  for select
  to authenticated
  using (true);

revoke all on table public.learning_submissions from public, anon;
grant select on table public.learning_submissions to authenticated;

-- 4. 학생 제출 전용 RPC 함수 (SECURITY DEFINER)
-- 학생은 회원가입 없이 학번 + 이름 + 비밀번호만 입력해 제출합니다.
-- 동일 (lesson_id, student_id) 재제출 시 비밀번호 해시가 일치할 때만 갱신합니다.
create or replace function public.submit_learning_log(
  p_lesson_id text,
  p_student_id text,
  p_student_name text,
  p_password text,
  p_payload jsonb
)
returns jsonb
language plpgsql
security definer
set search_path = public, extensions
as $$
declare
  v_lesson_id text := trim(coalesce(p_lesson_id, ''));
  v_student_id text := trim(coalesce(p_student_id, ''));
  v_student_name text := trim(coalesce(p_student_name, ''));
  v_password text := trim(coalesce(p_password, ''));
  v_existing public.learning_submissions%rowtype;
  v_new_count integer;
  v_updated_at timestamptz;
begin
  if v_lesson_id = '' then
    return jsonb_build_object('ok', false, 'error', 'INVALID_LESSON', 'message', '차시 식별자(lesson_id)가 비어 있습니다.');
  end if;
  if v_student_id = '' or char_length(v_student_id) > 30 then
    return jsonb_build_object('ok', false, 'error', 'INVALID_STUDENT_ID', 'message', '학번을 정확히 입력해 주세요.');
  end if;
  if v_student_name = '' or char_length(v_student_name) > 40 then
    return jsonb_build_object('ok', false, 'error', 'INVALID_NAME', 'message', '이름을 정확히 입력해 주세요.');
  end if;
  if char_length(v_password) < 4 or char_length(v_password) > 32 then
    return jsonb_build_object('ok', false, 'error', 'INVALID_PASSWORD_FORMAT', 'message', '비밀번호는 4자 이상 32자 이하로 입력해 주세요.');
  end if;

  select *
    into v_existing
    from public.learning_submissions
   where lesson_id = v_lesson_id
     and student_id = v_student_id;

  if not found then
    insert into public.learning_submissions (
      lesson_id,
      student_id,
      student_name,
      password_hash,
      payload,
      submit_count
    )
    values (
      v_lesson_id,
      v_student_id,
      v_student_name,
      crypt(v_password, gen_salt('bf')),
      coalesce(p_payload, '{}'::jsonb),
      1
    )
    returning submit_count, updated_at into v_new_count, v_updated_at;

    return jsonb_build_object(
      'ok', true,
      'mode', 'created',
      'submit_count', v_new_count,
      'updated_at', v_updated_at,
      'message', '작업 기록이 제출되었습니다.'
    );
  end if;

  -- 기존 제출 기록이 있으면 비밀번호 일치 여부 검증
  if v_existing.password_hash <> crypt(v_password, v_existing.password_hash) then
    return jsonb_build_object(
      'ok', false,
      'error', 'INVALID_PASSWORD',
      'message', '이미 제출된 학번입니다. 처음 제출할 때 사용한 비밀번호와 일치하지 않습니다.'
    );
  end if;

  update public.learning_submissions
     set student_name = v_student_name,
         payload = coalesce(p_payload, '{}'::jsonb),
         submit_count = submit_count + 1,
         updated_at = now()
   where id = v_existing.id
   returning submit_count, updated_at into v_new_count, v_updated_at;

  return jsonb_build_object(
    'ok', true,
    'mode', 'updated',
    'submit_count', v_new_count,
    'updated_at', v_updated_at,
    'message', '작업 기록이 최신 내용으로 다시 제출되었습니다.'
  );
end;
$$;

-- 익명(학생) 및 로그인 사용자(교사)에게 RPC 실행 권한 부여
revoke all on function public.submit_learning_log(text, text, text, text, jsonb) from public;
grant execute on function public.submit_learning_log(text, text, text, text, jsonb) to anon, authenticated;

-- 5. 교사 대시보드 / teacher.html 조회용 요약 뷰 (password_hash 제외)
create or replace view public.teacher_submission_summary
with (security_invoker = true) as
select
  id,
  lesson_id,
  student_id,
  student_name,
  submit_count,
  submitted_at,
  updated_at,
  payload
from public.learning_submissions
order by updated_at desc;

grant select on public.teacher_submission_summary to authenticated;
