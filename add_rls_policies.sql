-- Apex Coach AI — real Row Level Security policies (closing a documented-
-- but-never-built gap found during a full app audit)
--
-- profile_store.py's own module docstring has always claimed "RLS
-- policies added alongside this are defense-in-depth only" — but no such
-- policies existed anywhere in this repo's SQL until this file. Today,
-- that gap is not an ACTIVE vulnerability: every query in profile_store.py
-- already filters by coach_user_id correctly, and the anon key (the only
-- key that would ever actually be subject to RLS) is only ever used for
-- sign-in/sign-up in auth.py, never for querying athletes/sessions/teams
-- directly. This migration is what makes that claim true instead of
-- aspirational — a real second layer of defense if a future feature ever
-- queries these tables with anything other than the service key, or if
-- profile_store.py's own scoping ever has a bug.
--
-- SAFE TO RUN: the service key (SUPABASE_KEY, what profile_store.py
-- actually uses) bypasses RLS entirely, always, regardless of any policy
-- defined here — so enabling RLS and adding these policies changes
-- NOTHING about how the app currently behaves. It only starts enforcing
-- per-coach isolation for any request made with a non-service key.
--
-- coach_user_id = auth.uid() is compared via ::text on both sides rather
-- than assuming a specific column type — auth.uid() is always uuid, and
-- this repo's own athletes.coach_user_id column was added by a migration
-- (add_coach_scoping.sql) that was run directly in Supabase's SQL editor
-- and never committed here, so its exact declared type can't be confirmed
-- from this codebase. Casting both sides to text is correct regardless of
-- whether the underlying column is uuid or text.

alter table athletes enable row level security;
alter table teams enable row level security;
alter table sessions enable row level security;

drop policy if exists "coach_owns_athlete" on athletes;
create policy "coach_owns_athlete" on athletes
  for all
  using (coach_user_id::text = auth.uid()::text)
  with check (coach_user_id::text = auth.uid()::text);

drop policy if exists "coach_owns_team" on teams;
create policy "coach_owns_team" on teams
  for all
  using (coach_user_id::text = auth.uid()::text)
  with check (coach_user_id::text = auth.uid()::text);

-- sessions has no coach_user_id column of its own (it's scoped through
-- athlete_id) — the policy checks ownership via the parent athlete row,
-- same relationship profile_store.py's own _assert_owns_athlete() already
-- enforces in application code before any session read/write.
drop policy if exists "coach_owns_session_via_athlete" on sessions;
create policy "coach_owns_session_via_athlete" on sessions
  for all
  using (
    exists (
      select 1 from athletes
      where athletes.id = sessions.athlete_id
        and athletes.coach_user_id::text = auth.uid()::text
    )
  )
  with check (
    exists (
      select 1 from athletes
      where athletes.id = sessions.athlete_id
        and athletes.coach_user_id::text = auth.uid()::text
    )
  );
