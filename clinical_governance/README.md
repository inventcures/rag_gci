# Clinical knowledge governance

The module adds a clinical gateway and review workbench to the existing Palli Sahayak code. It does not replace clinical judgment or certify guideline concordance. Existing clinical studies and manuscript claims are unchanged.

## Try the synthetic workbench

Use the repository's Python environment, or install `requirements-governance.txt` in a separate environment with Python 3.10 or later.

```bash
venv/bin/python -m clinical_governance --demo --port 8012
```

Open [the local workbench](http://127.0.0.1:8012/governance). The command prints the path to a private temporary access-token file. The demo binds only to localhost and creates synthetic proposals without clinical approval. The token stays in browser memory, not local storage. Demo data and its in-memory audit key are disposable; do not use the demo database for clinical records.

## Enable the integration deliberately

The simple server, main server and legacy webhook server register `/governance` and `/api/governance/*`. The feature is off by default. No existing credentials or deployment environment were changed by this implementation.

| Mode | Behaviour |
| --- | --- |
| `off` | No governance database is opened. The workbench reports that configuration is required. Existing routes remain unchanged. |
| `parallel` | The authenticated governance API works alongside explicitly legacy routes. It does not make legacy vector, graph, phone or browser-voice responses clinically governed. |
| `strict` | The HTTP boundary blocks legacy routes, including WebSockets, except the health route. Direct calls to the two legacy RAG query methods are also blocked. Use authenticated sessions and pinned releases through the governance API. |

Strict mode intentionally does not advertise the existing Bolna/Gemini audio paths as governed. They need a separately tested session and transport adapter before being enabled under this boundary. Do not remove the block merely to make a demonstration appear functional.

Required configuration for `parallel` or `strict`:

| Environment variable | Meaning |
| --- | --- |
| `CLINICAL_GOVERNANCE_MODE` | `parallel` or `strict` |
| `CLINICAL_GOVERNANCE_DB` | Private database path; default `data/clinical_governance/governance.sqlite3` |
| `CLINICAL_GOVERNANCE_AUDIT_KEY` | Independent high-entropy secret of at least 32 bytes; keep outside Git |
| `CLINICAL_GOVERNANCE_PRINCIPALS_JSON` | Operator-provisioned array of token hashes, actor IDs, tenants, roles and permitted purposes |

Principal shape (the placeholder is not a working credential):

```json
[
  {
    "token_sha256": "<SHA256 of a separately generated high-entropy bearer token>",
    "actor_id": "reviewer-account-id",
    "tenant_id": "institution-id",
    "roles": ["editor", "clinician"],
    "purposes": ["information", "quality_review"],
    "clinical_authority": true
  }
]
```

Set `clinical_authority` only after the institution has authorized that reviewer. The application authenticates the provisioned account; it does not verify medical qualifications. Publisher, operator, auditor, requester, billing, evaluator, language reviewer and usability reviewer are separate roles. A mobile self-registration role cannot grant governance privileges.

Use HTTPS and appropriate network controls outside localhost. Provision separate tokens per person or service and rotate them through the operator configuration. The current adapter uses provisioned bearer credentials, not an institutional OIDC directory. Audit-key rotation requires a migration that preserves verification of old events; do not casually replace the key for an existing database.

## Author, review and publish

First, import an authorized source. Original bytes and normalized text are stored separately with hashes. Text ingestion preserves UTF-8 content. PDF ingestion records page and text-block coordinates, detected table cells, and explicit gaps for images, scans and unresolved semantics. Scanned PDFs are not silently OCRed. Clinical table/header/footnote interpretation still needs review.

Second, author a recommendation with exact evidence spans, explicit applicability conditions, qualifiers, exceptions, alternatives, search keywords and patient-friendly wording in specified languages. Each edit creates an immutable new revision. The UI compares evidence and interpretation and can show prior/current revisions. Explanations are part of the reviewed revision, not independent unreviewed text.

Third, a provisioned clinician records a decision for that exact revision, with a rationale and explicit evidence/exception/coverage checks. A publisher then creates an immutable release. Clinical approval, publication, automatic applicability, request review and patient-care action authorization remain separate. No API operation authorizes autonomous treatment actions.

Runtime eligibility checks current source and recommendation versions, current approval identity, exact evidence, recorded rights, purpose, expiry, blocking conflicts, release membership and revocation. Reapproving a later recommendation never revives an old release. Even a new approval of the same revision requires republishing when the approval identity changes. Withdrawal and correction history are retained.

The applicability evaluator supports bounded `all`, `any`, `not`, equality, membership and numeric comparison expressions. A missing declared variable remains unknown. The evaluator does not determine whether an author omitted a clinically necessary qualifier. Clinical semantic completeness remains a qualified-review responsibility.

## KL4A adapter

The adapter reads the canonical Markdown/frontmatter subset from the [pinned KL4A bundle specification](https://github.com/CogniSwitch/KL4A/blob/f66495923ebaea05bfe08f947caeb6510b8b1aea/docs/OKF_BUNDLE_SPEC.md). It targets OKF `0.2` and KL4A profile `0.2.0`, not universal OKF conformance.

Import the ZIP with `index.md` and `manifest.yaml` at its root. ZIPs are read in memory with path, symlink, size, duplicate-entry and expansion checks. YAML aliases and duplicate keys are rejected. Original bundle bytes, unknown frontmatter fields and upstream review labels are retained. `.sopkb` caches and agent instructions do not supply clinical authority.

Imported knowledge is an unreviewed candidate. Attach its evidence to a locally authored recommendation, define its clinical scope, then obtain local approval. A displayed upstream `approved` or `edited` label does not approve the local item. Offsets index code points in normalized Markdown; the implementation preserves explicit UTF-8 and UTF-16 conversions for browser use.

Updating an existing bundle requires selecting the latest local import explicitly. Its source-family revisions then invalidate older dependent evidence. Untrusted bundle IDs cannot silently replace existing sources. Removal of an earlier source requires explicit withdrawal rather than silently dropping it during an update.

Source permissions cover the named purpose and expiry. External processing defaults to false. Enabling it requires a list of named permitted processors. Permission records are operator assertions, not legal verification. Do not upload licensed guidelines unless storage, processing and any redistribution are authorized.

## Governed requests

Start a session with a release and purpose, then send the question, explicit context, desired language, session epoch and unique request ID. Each new request advances the input sequence. A later request or interruption invalidates older background results. Idempotent retries revalidate evidence instead of blindly returning a cached answer.

Candidate selection currently uses clinician-authored keywords with Unicode-aware tokenization. It is a conservative, inspectable baseline, not a validated semantic clinical retriever. No match yields an explicit gap. Candidate retrieval never falls back to the full bundle or legacy indexes.

The five request-review modes are off, sampled, user-requested, policy-triggered and required. Policy belongs to the operator, not an untrusted caller parameter. Callers may request more review, but cannot weaken the configured mode. Review-off requests still receive applicability checks and durable auditing. Request-review approval requires an authenticated clinician's context-review attestation.

The answer path returns the reviewed wording for the requested language, with claim/evidence links. It does not invoke an LLM, invent missing advice, or silently translate into an unreviewed language. Answer preparation is not proof of delivery. Client display/playback reports are recorded separately; late reports about stale content are flagged rather than erased. No delivery flag proves comprehension.

## Audit and privacy

Every governed operation and handled denial shares a SQLite transaction with a minimized audit record. Audit failure prevents the action from being returned as successful. Events use a per-tenant chained HMAC and append-only triggers. Source/recommendation versions and original bytes have immutable records. The database and its journal files are private files.

Request text and context are fingerprinted with a secret key rather than copied into operational audit. Retaining raw query/context requires both a nonzero operator retention policy and explicit request consent. Retained payloads have their own expiry and access rules. `purge_payloads` deletes expired payloads and records the count. Review rationales, source content and clinical recommendations are retained as governed records, so do not put patient identifiers or credentials into those fields.

An auditor can verify the stored chain, but this is not an independently witnessed checkpoint or a complete compliance system. A privileged administrator controlling both storage and key can undermine local protection. Production work still needs secure backup, off-host integrity checkpoints, incident handling, institutional retention rules and an appropriate source-content erasure policy. No production privacy or clinical safety certification is claimed.

## INR ledger and paid transforms

Operators configure session/monthly limits in integer paise and enter versioned provider rates with references and expiry. No real prices are supplied. Reservations are atomic across concurrent sessions; unknown rates cannot permit a paid call. Uncertain or cancelled calls remain reserved until accounted for. Reconciliation records actual charges, including overruns, without changing the original reservation. The budget period is the UTC month in which work was reserved, not an asserted invoice billing month.

`paid.run_paid_transform` is a server-side adapter helper for a bounded transform of approved wording, such as TTS. It checks named processor permission and request eligibility before external processing, reserves the bounded quote, and avoids automatic repeat execution of the same reservation. The callback receives only the approved text. Timeout/output bounds are enforced, and the adapter must enforce its quoted provider units and retry limits. Actual costs are reconciled before checking whether a now-stale result must be discarded.

No existing provider was activated or purchased. The helper's tests use async fakes, and no public API accepts a user-supplied callable. A real speech adapter needs explicit provider/rate configuration, permitted data processing and independent speech validation. A transport test or a rate reservation is not proof of clinical correctness or a guarantee about an external invoice.

## API and verification

Use a bearer token with the `/api/governance` endpoints. `POST /ingest` accepts raw PDF/text/ZIP bytes and a JSON `X-Source-Metadata` header. For Indic metadata, URI-encode the JSON and set `X-Source-Metadata-Format: uri`. Other operations accept JSON through `POST /{operation}`. The implementation's `OPERATIONS` allowlist defines the public surface. Internal transport and provider operations are not exposed through that generic endpoint.

```bash
venv/bin/python -m pytest tests/test_clinical_governance.py tests/test_governance_ingestion_budget.py tests/test_governance_api.py tests/test_governance_paid.py -q
ruff check clinical_governance --select F
node --check clinical_governance/web/app.js
```

The suite uses temporary databases and synthetic protocols. Validation records distinguish engineering, clinical, language and usability work. A human review cannot be marked performed by a software test, and an engineering pass does not approve a clinical release. Source changes and revocations remain visible when checking earlier validation records.

Pending work before patient-facing use includes qualified clinical/language review, representative usability testing, more complete clinical extraction, a validated semantic retriever, institutional identity integration, live voice adapter benchmarking and production governance/security review. No deployment, clinical approval, new patient study or manuscript update was performed as part of this code change.
