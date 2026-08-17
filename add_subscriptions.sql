-- Apex Coach AI — subscription tiers + manual payment verification
--
-- No merchant/API payment account exists yet (personal JazzCash/Easypaisa/
-- bank accounts only, confirmed 2026-08-17) — so this is a MANUAL
-- verification flow, not an automated checkout: a coach picks a tier,
-- is shown where to send payment, submits their own transaction
-- reference, and an admin approves/rejects it by hand (checking the
-- actual JazzCash/Easypaisa/bank transaction history themselves). This
-- is a legitimate, common pattern for an early-stage business without
-- merchant API access yet — swap for a real gateway integration
-- (Safepay or similar) later without changing the tier/usage-gating
-- logic that reads from `subscriptions`, only how it gets written to.
--
-- period_start/used_this_period (2026-08-17, real pricing sheet):
-- every paid tier is priced as "N analyses PER MONTH," not per day —
-- a rolling 30-day usage-allowance window, tracked independently of
-- the free tier's own daily demo_usage table (unchanged, still daily).
-- An annual subscriber still gets a FRESH allowance every 30 days
-- within their year, not one lump sum for the whole 12 months.
--
-- Same security model as every other user-scoped table in this app:
-- profile_store.py's service key bypasses RLS for the app's own writes;
-- RLS here is defense-in-depth (see add_rls_policies.sql's own note),
-- not the primary enforcement.

create table if not exists subscriptions (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null unique,
    tier text not null default 'free',      -- 'free' | 'starter' | 'professional' | 'elite' | 'institutional'
    status text not null default 'active',   -- 'active' | 'expired' | 'cancelled'
    currency text,                            -- 'pkr' | 'usd' — null for free tier
    expires_at timestamptz,                   -- null for free tier (never expires)
    period_start timestamptz,                 -- start of the CURRENT 30-day usage-allowance window
    used_this_period integer not null default 0,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_subscriptions_user_id on subscriptions(user_id);

create table if not exists payment_submissions (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null,
    user_email text not null,              -- denormalized for admin review without a join
    tier_requested text not null,          -- 'starter' | 'professional' | 'elite' | 'institutional'
    billing_period text not null,          -- 'monthly' | 'annual'
    currency text not null,                -- 'pkr' | 'usd'
    amount numeric not null,               -- in the submission's own currency
    payment_method text not null,          -- 'jazzcash' | 'easypaisa' | 'bank_transfer'
    transaction_reference text not null,
    submitted_at timestamptz not null default now(),
    status text not null default 'pending', -- 'pending' | 'approved' | 'rejected'
    reviewed_by text,
    reviewed_at timestamptz,
    admin_notes text
);

create index if not exists idx_payment_submissions_status on payment_submissions(status);
create index if not exists idx_payment_submissions_user_id on payment_submissions(user_id);

alter table subscriptions enable row level security;
alter table payment_submissions enable row level security;

drop policy if exists "coach_owns_subscription" on subscriptions;
create policy "coach_owns_subscription" on subscriptions
  for select
  using (user_id::text = auth.uid()::text);

-- Regular coaches can submit and view their own payment claims, but
-- cannot approve/reject their own submission (no UPDATE policy for
-- them) — status changes only ever happen through the admin panel,
-- which uses the service key and so bypasses RLS entirely anyway; this
-- policy exists so a direct anon-key query could never self-approve.
drop policy if exists "coach_submits_own_payment" on payment_submissions;
create policy "coach_submits_own_payment" on payment_submissions
  for insert
  with check (user_id::text = auth.uid()::text);

drop policy if exists "coach_views_own_payment" on payment_submissions;
create policy "coach_views_own_payment" on payment_submissions
  for select
  using (user_id::text = auth.uid()::text);
