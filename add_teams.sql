-- Apex Coach AI — Teams/Academy roster schema
-- Adds real multi-bowler roster support: a coach can group athletes under
-- a team/academy, matching the same coach-scoping security model already
-- used for athletes/sessions (service key bypasses RLS; coach_user_id is
-- the real access control, enforced in profile_store.py's queries).

create table if not exists teams (
    id uuid primary key default gen_random_uuid(),
    name text not null,
    coach_user_id uuid not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_teams_coach_user_id on teams(coach_user_id);

-- Nullable: an athlete doesn't have to belong to a team. One team per
-- athlete for now (not a join table) — matches how a real academy roster
-- actually works, and keeps the query model simple. Revisit only if a
-- real need for multi-team membership shows up.
alter table athletes add column if not exists team_id uuid references teams(id) on delete set null;

create index if not exists idx_athletes_team_id on athletes(team_id);

-- Bowler Profile page: optional photo + free-text coaching notes per athlete.
-- Both nullable — a profile with neither still works, it just shows a
-- placeholder avatar and no notes card.
alter table athletes add column if not exists photo_url text;
alter table athletes add column if not exists notes text;
