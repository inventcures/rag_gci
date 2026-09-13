# Clinical governance implementation

User authorization: implement the latest KL4A synthesis on the latest code and periodically commit/push to the configured GitHub repository. No deployment requested.

Baseline: `51d307f`, matching origin/main at initial fetch. Work branch: `codex/clinical-knowledge-governance`. Preserve pre-existing admin-port edits in `simple_rag_server.py`, database files, caches, manuscripts and other untracked artifacts. Stage only task-owned changes.

- [ ] Inspect current routes, authentication, retrieval, voice, costs and upstream interchange contracts.
- [x] Build versioned source/recommendation/release store, strict applicability and mandatory durable audit; 31 synthetic regression tests pass. Checkpoint commit/push follows.
- [ ] Add evidence-preserving ingestion and pinned KL4A adapter with unknown-field preservation, approval/revocation/review workflows and INR ledger; tests; checkpoint.
- [ ] Add authenticated governance API and accessible source/revision comparison UI; integrate guarded retrieval into existing server without silently bypassing policy; checkpoint.
- [ ] Add validation workflow, end-to-end synthetic scenarios, security/failure tests, documentation and final verification; checkpoint.

Constraints: no clinical approvals fabricated; no licensed guideline uploads or paid provider calls during tests. Existing production routes remain explicitly legacy until governance is configured; governed requests never fall back to legacy evidence. Human request review is configurable, audit mandatory. All model/provider claims are checked against current code or primary documentation.
