# Apex Coach AI — Product & Company Brief

*Prepared for use by another AI assistant running a marketing campaign. This is a factual reference document, not marketing copy itself — pull from it to write copy, don't quote it verbatim as finished material. A "Guardrails" section at the end lists claims that must NOT be made; read that section before writing anything customer-facing.*

Last updated: 2026-08-17.

---

## One-paragraph summary

Apex Coach AI is a biomechanical analysis platform for cricket — fast bowling and batting technique — that turns a single phone video into measurable, trackable data. It's built by a working cricket academy (Strikers Den Sports Academy, Karachi, Pakistan) to solve a real problem the founder hit as a coach: technique assessment in cricket runs almost entirely on a coach's eye, which is valuable but subjective and doesn't leave a trail across months of training. The product is a Streamlit web app: a coach uploads a video, the system runs pose estimation and computer vision, and returns a structured report with named biomechanical metrics, color-coded severity zones, an AI-generated coaching narrative, and prescribed drills — plus a PDF export.

## Who built it, and why that matters for credibility

**Shoaib Nazar** — founder, ICC Level 2 Certified Coach, founder of Strikers Den Sports Academy.
- **As a player**: competitive club cricket in Karachi, captained his zone at Under-15/17/19 as an opening batsman — real, lived experience of facing fast bowling, not just an outside observer of the sport.
- **As an operator**: 18 years across Pakistan, the UAE, and Singapore in telesales, real estate consulting, and technology — founded a digital agency (AOS Formula), co-built the event platform Oyee.pk, and served as Assistant Vice President at Riztech Pvt Ltd running brand, engineering, and digital operations. This is the background that shaped the product's insistence on measurable, verifiable output rather than vague claims.
- **As a coach**: returned to cricket as an ICC Level 2 Certified Coach, founded Strikers Den Sports Academy, and went on to lead sport across six private school campuses, authoring two coaching frameworks (*Built to Win*, a Cambridge PE curriculum for Grades 3–10; *Built to Perform*, an elite cricket coaching manual).

The product exists because the founder personally felt the gap it fills — this is a domain-expert-built tool, not a generic sports-tech play bolted onto cricket.

## What the product actually does today

### 1. Bowling Analysis (the original, most mature module)
- Pose extraction via MediaPipe (33 body landmarks tracked per frame) from an uploaded video.
- Automatic detection of three key delivery events: Back Foot Contact, Front Foot Contact, Ball Release.
- Five core biomechanical metrics computed from that landmark data: Lead Knee Bracing Angle, Hip-Shoulder Separation, Trunk Lean Deflection, Release Height Ratio, Head Stability Variance.
- Each metric classified into Optimal / Acceptable / Critical zones, shown consistently across the on-screen report, the PDF export, and the AI coaching narrative (one shared source of truth, never three different opinions).
- AI-generated technical assessment plus three prescribed training drills, grounded in the same classification the coach already sees.
- Run-up analysis: stride count, pacing consistency, foot-strike pattern (heel/midfoot/forefoot).
- Release-arm speed estimate — but only once the camera is calibrated against a known real-world distance (e.g. stump width); without calibration, no number is shown at all (see Honesty section).
- Athlete history: sessions save per named athlete, so a coach can track one bowler's technique across multiple sessions over time.

### 2. Batting Analysis (added ~August 2026, same coaching engine)
- Works from any filming angle (side-on or front-on/rear-on) — auto-detected from the footage, not a rigid single camera position requirement.
- Seven core metrics: head movement, front-foot alignment, weight transfer, downswing plane, top-elbow control, front-knee flexion, X-factor separation.
- Shot-relative foot alignment: a cover drive and a straight drive have different "correct" foot positions, and the scoring knows the difference rather than applying one generic rule.
- A dedicated "falling-over" alert: flags when a batter's head and front foot are both drifting toward the danger side of the delivery line — a named, recognizable coaching fault, not just raw numbers.
- Same AI narrative + drills + honesty standard as bowling analysis.

### 3. Ball Tracking (in active development, NOT yet a finished customer-facing feature)
- A seeded computer-vision tracker (YOLO-based object detection + a physics-informed local search) that follows the ball's trajectory forward from a coach's confirmed click on one frame.
- Renders a trajectory overlay on the video.
- **Current real status**: works reliably on clean, single-subject footage (validated against several real, human-labeled clips). Struggles on cluttered/crowded scenes with multiple people or confusing background objects, where it can lock onto the wrong object. This is being actively worked on. Ball speed and a synthetic 3D pitch/stump overlay (the kind FullTrack AI shows) are NOT built yet — they depend on camera calibration precision work that has real, documented accuracy challenges at typical phone-video resolutions.

### Platform-level features
- Per-athlete session history, stored in Supabase.
- Coach accounts with a daily free-usage limit (currently 2 free analyses/day per account) — the free tier that any paid tier will sit alongside.
- Admin allowlist mechanism for gating in-development features (currently used for ball tracking).
- PDF report export.
- Dark-themed, mobile-friendly Streamlit UI.

## How it works, technically (for context, not for verbatim marketing copy)

- **Pose estimation**: Google's MediaPipe, tracking 33 body landmarks per frame from ordinary phone video — no special markers, suits, or lab equipment.
- **Ball detection**: a custom-trained YOLO (You Only Look Once) object detector, fine-tuned on hand-labeled cricket-ball footage collected specifically for this project (coach-clicked ground truth, not a generic pretrained "sports ball" class).
- **Speed estimation**: requires the coach to calibrate the camera once (click two points of a known real-world distance, e.g. stump width) — without that, the app explicitly says "not available" rather than guessing.
- **Backend**: Supabase (Postgres + auth + storage).
- **Frontend/app**: Streamlit (Python).
- Runs from a single uploaded phone video — no fixed-tripod rig, no dedicated recording setup required for the core bowling/batting analysis (ball tracking's harder cases do better with cleaner, less crowded footage, same as any computer-vision system).

## Competitive position

The most relevant named competitor is **FullTrack AI** (and similar broadcast-style tracking systems), which requires a fixed tripod, elevated ~1.5m, positioned ~4m behind the bowler's stumps with both sets of stumps visible — a dedicated filming setup most club/academy coaches don't have and won't set up for daily training. Apex Coach AI's bowling and batting analysis work from the kind of handheld footage a coach is already shooting in the nets. Ball tracking specifically is the one area where FullTrack's fixed-camera approach currently has a real advantage (more consistent geometry for tracking and calibration) — Apex Coach AI does not yet match FullTrack's ball-tracking reliability or its speed/pitch-map output, and marketing copy must not imply otherwise (see Guardrails).

The deeper differentiator, independent of any single feature, is the **honesty design principle** (see below) — most sports-tech tools present a confident number regardless of input quality; this one is built to say "I don't know" when it doesn't know.

## Target audiences

1. **Cricket academies** (B2B) — like the founder's own Strikers Den Sports Academy. Value prop: give every coach on staff a consistent, repeatable way to assess technique, not dependent on any one coach's individual eye; track a whole roster of students over a season; a credible, quantified addition to what parents are already paying the academy for.
2. **Individual coaches** (B2C/prosumer) — freelance or club coaches without academy infrastructure. Value prop: professional-grade technique breakdown and a PDF report to hand to a player/parent, without lab equipment or a videographer.
3. **Parents of young cricketers** — likely a secondary/downstream audience reached through coaches and academies rather than directly, but relevant for messaging: objective, trackable proof of a child's technical progress over a season.
4. **Investors** (for fundraising conversations) — the founder is actively evaluating commercialization and has expressed interest in raising investment after a product demo. Relevant framing: domain-expert founder (coach + 18 years operator background), real academy already using it, a genuine technical moat in the ball-tracking/pose-estimation data pipeline being built up through real coach-labeled data (not bought/generic training data), and a large addressable market (cricket is played seriously in Pakistan, India, and across the Commonwealth).

## Business model (as of this brief)

- Currently: free tier only, capped at a small number of analyses per day per account, used for internal testing and demo coach accounts.
- **In progress**: monthly/annual paid subscription tiers are being built (decision made 2026-08-17) — exact tier names, limits, and pricing are not finalized yet; don't state specific prices in marketing copy until confirmed.
- Payment processing: still being decided. The business is Pakistan-based, which rules out Stripe for direct payouts — a Pakistan-compatible processor (e.g. a local gateway, or a merchant-of-record service) is the likely path. Not yet live.

## The "Honesty, By Design" principle — the core brand value, use it deliberately

This is stated explicitly on the app's own About page and is a real, enforced engineering principle, not just a slogan:
- If tracking quality is too poor to trust a reading, the report says so instead of showing a number anyway.
- If a camera isn't calibrated, no speed number is shown — ever, for any reason.
- Release-arm speed is explicitly labeled as an estimate of the WRIST's speed near release, not the ball's actual speed (a radar gun measures the ball, which leaves the hand faster than the hand itself moves) — this distinction is deliberately surfaced, not hidden.
- Every classification a coach sees — on-screen, in the PDF, in the AI narrative — comes from one shared set of reference ranges, so the tool can never contradict itself.

This is genuinely a strong, differentiated marketing angle: **"the tool that tells you when it doesn't know," in a category full of tools that always show a confident number.**

---

## Guardrails — claims that must NOT appear in marketing copy generated from this brief

1. **Do not claim "lab-grade" accuracy, or any specific percentage accuracy (e.g. "99% accurate").** This has been explicitly rejected internally before. Use language like "measurable," "repeatable," "data-backed" instead of unverifiable precision claims.
2. **Do not claim the app measures actual ball speed.** It measures release-ARM/wrist speed as an estimate, and only when the camera has been calibrated. Never say "ball speed" or "bowling speed" as something the app currently outputs as a headline feature.
3. **Do not claim a 3D pitch map, virtual/synthetic stumps, bounce-point prediction, or any FullTrack-AI-style broadcast visualization exists.** It does not exist yet. Ball tracking today is a 2D trajectory overlay on the original video, and only reliable on clean footage.
4. **Do not claim ball tracking works on any footage, or claim it as a finished, reliable, generally-available feature.** It is real and in active development, currently gated to internal/admin testing, proven on clean single-subject clips, with known real failure modes on crowded/cluttered footage.
5. **Do not invent specific pricing.** Subscription tiers are in progress but not finalized as of this brief.
6. **Do not claim this replaces coaching judgment.** The consistent internal framing is "a complement to expert coaching judgment, not a replacement for it" — keep that framing in any copy.
7. **Do not claim broad multi-sport support.** This is a cricket-specific platform (fast bowling + batting technique) — no other sports are supported.
8. **When in doubt about whether a technical claim is accurate, prefer the more conservative version**, or flag the uncertainty back to the founder rather than guessing — this matches how the product itself is built.
