# Implementation verification

The implementation started from `51d307f`, which matched origin/main at the initial fetch. Changes are on `codex/clinical-knowledge-governance`. The existing local admin-port edit and generated/private artifacts were preserved and excluded from commits.

## Results

- 69 new governance tests passed. They cover source/evidence integrity, Unicode offsets, publication and revision checks, all request-review modes, no-match/unknown behaviour, tenant/role boundaries, request retention, stale input and delivery, audit failure/tampering, ZIP safety, PDF maps, import updates, INR concurrency/reconciliation, paid transforms and validation records.
- 28 existing Retell and PageIndex configuration tests passed.
- Ruff undefined/unused-name checks passed for the new module and tests. Python formatting checks and JavaScript syntax checks passed.
- The modified server entry points compiled. The full provider-dependent application/integration suite was not run; no real provider calls, patient records or licensed guideline files were used.

Command:

```bash
venv/bin/python -m pytest tests/test_clinical_governance.py tests/test_governance_ingestion_budget.py tests/test_governance_api.py tests/test_governance_paid.py tests/test_retell_config.py tests/test_pageindex_config.py -q --tb=short
```

Observed result: 97 passed. Existing pytest-asyncio configuration and PyMuPDF SWIG deprecation warnings remain; they did not fail the suite.

## Browser check

The localhost synthetic workbench was inspected in the in-app browser. Login, source/interpretation comparison, expanded status reasons, editing to a new proposed revision and prior/current comparison worked. The demo account had no clinical authority, and no clinical approval was seeded. The UI preserved the distinction between passing automated checks and pending clinical review.

Desktop rendering was inspected. Responsive CSS is implemented, but representative mobile usability and accessibility conformance were not independently validated. No human participant study or native-speaker validation is claimed.

## Fail-closed checks

Strict mode rejects legacy HTTP routes and WebSockets. A governed retrieval never calls a legacy index or model. Source, recommendation and release revocations are checked before answer preparation. A changed request invalidates older background output. A late report of stale playback is retained with a possible-unsafe-delivery flag, not treated as authorization or proof of comprehension.

Paid-transform tests use in-process fakes. They prove reservation-before-call behaviour, named-processor permission, no automatic replay, bounded timeout/output, reconciliation of known charges and rejection of stale output. They do not establish provider prices, invoice correctness, speech quality or live latency.

## Remaining prerequisites

Enablement is opt-in. An operator must provision principals and an independent audit key, confirm source rights, and choose review and retention policies. Actual clinical approval and clinical/language/usability validation require authorized humans.

The KL4A adapter is a pinned import subset. PDF extraction preserves evidence but does not automatically validate clinical tables or exceptions. The keyword retriever is an inspectable baseline, not validated clinical semantic retrieval. Strict live telephony/browser-audio adapters, institutional identity integration, off-host audit witnesses and production retention/erasure operations need further integration and validation before patient-facing use.
