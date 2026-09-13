# Clinical governance implementation

User authorization: implement the latest KL4A synthesis on the latest code and periodically commit/push to the configured GitHub repository. No deployment requested.

Baseline: `51d307f`, matching origin/main at initial fetch. Work branch: `codex/clinical-knowledge-governance`. Preserve pre-existing admin-port edits in `simple_rag_server.py`, database files, caches, manuscripts and other untracked artifacts. Stage only task-owned changes.

- [x] Inspect current routes, authentication, retrieval, voice, costs and upstream interchange contracts.
- [x] Build versioned source/recommendation/release store, strict applicability and mandatory durable audit; 31 synthetic regression tests pass. Checkpoint commit/push follows.
- [x] Add evidence-preserving ingestion and pinned KL4A adapter with unknown-field preservation, approval/revocation/review workflows and INR ledger; 49 synthetic tests pass. No clinical semantic extraction or human validation is fabricated.
- [x] Add authenticated governance API and source/revision comparison UI; integrate strict boundary into all server entry points and direct legacy queries. Desktop browser check and 60 synthetic tests pass. No human usability validation claimed.
- [x] Add validation workflow, end-to-end synthetic scenarios, security/failure tests, documentation and final verification. Final focused suite: 97 tests pass (69 new governance tests and 28 existing configuration tests). Final checkpoint follows.

Constraints: no clinical approvals fabricated; no licensed guideline uploads or paid provider calls during tests. Existing production routes remain explicitly legacy until governance is configured; governed requests never fall back to legacy evidence. Human request review is configurable, audit mandatory. All model/provider claims are checked against current code or primary documentation.

Delivered software scope: local governance workbench, strict/parallel integration, source and release lifecycle, PDF/text/KL4A import staging, optional request review, mandatory audit, stale-input rejection, INR ledger and bounded paid-transform adapter, validation records and regression suite. See `clinical_governance/README.md` for configuration and remaining clinical/production prerequisites. No deployment or live provider was enabled.
