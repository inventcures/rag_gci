---
description: Show every ticket with status, blockers and acceptance progress
argument-hint: "[ticket-number|all|detail]"
---

Read the Palli Sahayak ticket set and report status. Do not guess: read the files.

Tickets live in the repository at `docs/tickets/*.md`. They were moved out of
`/home/tp53/.scratch/` on 2026-10-04, because outside the repository they were not
version controlled and a machine reset would have taken them.

Search in this order and use whichever exists:

1. `<repo>/docs/tickets/*.md`
2. `/home/tp53/.scratch/android-app-v1/issues/*.md`, a pre-move location that may
   still hold stale copies. If both exist, trust the repository and say so, because
   two copies of a status file disagree exactly when it matters.

If neither exists, say so plainly and point at `docs/HANDOVER.md`, which carries the
ticket list inline.

## Argument

- no argument, or `all` — one row per ticket: number, title, status, blockers
- a number such as `3` — full detail for that ticket only
- `detail` — every ticket with its full acceptance-criteria checklist

## How to read each ticket

Parse these fields from the frontmatter block at the top of each file:

- `**Status:**` — treat as complete only when every acceptance box is ticked
- `**Blocked by:**` — list the numbers
- the `- [ ]` / `- [x]` lines are the acceptance criteria

Count the boxes. A ticket claiming `complete` with unticked boxes is a contradiction and
should be reported as such rather than repeated.

## Output

Start with a summary table. Keep it scannable: number, title, status, blockers, criteria
progress as `3/8`.

Then handle the argument:

- detail mode, or a single ticket: expand that ticket with its blockers, its acceptance
  criteria and their tick state, its Verification notes, and its Known limitations.
- otherwise: list the frontier, meaning tickets whose blockers are all complete. That is
  what can start now.

Close with anything that looks wrong: a status that contradicts its ticked boxes, a
blocker that no longer exists, or a ticket whose blockers are all done but which was
never started.

Keep it short. The table is the point; the rest is supporting detail. Do not restate
criteria that are ticked.
