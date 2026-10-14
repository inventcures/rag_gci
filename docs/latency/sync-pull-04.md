# Ticket 04: sync pull and patient list, before and after

`python scripts/benchmark_sync_pull.py --patients 1000` (1,000 patients, 20
observations each, ten app users, one user timed). "Before" is the same script
run against `main`'s `personalization/longitudinal_memory.py` and
`sync/delta_tracker.py`. Offline, local disk, one run each; the spans are in the
benchmark trace files.

| call | before ms | after ms | patients returned (before / after) |
|---|---|---|---|
| first sync (`last_sync_at=0`) | 2,240 | 96 | 1,000 / 100 |
| routine sync, nothing changed | 2,032 | 7 | 1,000* / 0 |
| patient list | 656 | 6-7 | 1,000 / 100 |

\* The benchmark backdates file times but not the stored `updated_at`, so the
old code, which compared `updated_at`, still returned every patient. In
production the old routine sync returned only changed patients, but parsed every
file three times to find them; the after column parses none.

Where it went: three full scans of every file per pull, each JSON-parsed and
(for observations) rebuilt into objects, became one pass that skips files on
their modification time and keeps a per-file owner summary keyed on
modification time and size.
