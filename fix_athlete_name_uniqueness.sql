-- Apex Coach AI — fix athlete name uniqueness to be per-coach, not global
--
-- ROOT CAUSE of "duplicate key value violates unique constraint
-- athletes_name_key": the original supabase_schema.sql defined
-- `name text not null unique` — a GLOBAL uniqueness rule across every
-- coach's athletes, predating coach_user_id scoping being added to the
-- app. profile_store.py's get_or_create_athlete() already looks up an
-- existing athlete scoped by coach_user_id (so two coaches can each have
-- their own "Mustafa J"), but the database's own constraint never
-- matched that — the first time a second coach (or the same coach after
-- the row moved/changed) used a name that already existed for ANYONE,
-- the app's scoped SELECT correctly found nothing, tried to INSERT a new
-- row, and Postgres rejected it because `name` alone still had to be
-- unique across the whole table.
--
-- This replaces that with a composite constraint: unique per (name,
-- coach_user_id) — the same athlete name is fine across different
-- coaches, and still prevented from being duplicated within one coach's
-- own roster (defense-in-depth backing get_or_create_athlete's own
-- lookup-before-insert check).

alter table athletes drop constraint if exists athletes_name_key;
alter table athletes add constraint athletes_name_coach_user_id_key unique (name, coach_user_id);
