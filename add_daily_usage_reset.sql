-- Apex Coach AI — daily reset for the demo usage limit
--
-- demo_usage previously tracked a pure LIFETIME cap: used_count only ever
-- went up, free_limit was a one-time ceiling, and there was no date
-- column at all — a coach who used up their free_limit was capped
-- forever, not "for today." Adding usage_date turns this into a real
-- daily allowance: usage_limits.py now resets used_count back to 0
-- whenever it sees usage_date isn't today, before checking or
-- incrementing it.
--
-- DEFAULT current_date backfills existing rows to "today" rather than
-- leaving usage_date null — this is a genuinely correct value for them
-- (their existing used_count IS today's count as far as this migration
-- is concerned), not a guess standing in for missing data.

alter table demo_usage add column if not exists usage_date date not null default current_date;
