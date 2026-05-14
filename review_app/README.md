# Review Console — 6-doctor RAG output grading

A small FastAPI + Jinja2 web app for the Palli Sahayak Phase 2 clinical validation
study. Six palliative care doctors log in with a PIN and grade RAG-generated
answers against 40 difficult-scenario vignettes using the v53 rubric.

## Run locally

```bash
cd review_app
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8005
```

Then visit <http://localhost:8005>.

## Deploy to Railway

The repo root contains `railway.toml` (service config) and `nixpacks.toml`
(build plan). Railway autodetects both.

1. Push the repo to GitHub (already at `inventcures/rag_gci`).
2. Railway dashboard → **New Project** → **Deploy from GitHub repo** →
   select `inventcures/rag_gci`.
3. Railway reads `railway.toml` + `nixpacks.toml`, installs
   `review_app/requirements.txt`, and starts uvicorn.
4. **Set environment variables** in the service's **Variables** tab:
   ```
   SECRET_KEY=<run: openssl rand -hex 32>
   REVIEW_DATA_DIR=/data
   ```
5. **Attach a Volume** (Service → Settings → Volumes → + New Volume):
   ```
   Mount path: /data
   Size:       1 GB
   ```
   This persists reviews + session secret across deploys.
6. **Generate a public domain**: Service → Settings → Domains →
   **Generate Domain** (free `*.up.railway.app`) or add a custom domain.
7. After ~3 min the app is live. Distribute URL + PINs to the 6 reviewers.

### Environment variables

| Variable | Set by | Purpose |
|----------|--------|---------|
| `SECRET_KEY` | You (Variables tab) | HMAC key for session cookies — required |
| `REVIEW_DATA_DIR` | You → `/data` | Persistent volume mount; reviews + session secret |
| `PORT` | Railway auto-injects | Web port (passed to uvicorn) |

### Without a Volume (cheaper, but ephemeral)

If you skip the Volume attachment, reviews are lost on every restart.
Download `/export.csv` frequently to capture reviews before any redeploy.

## Default PINs

The first time the app boots it seeds `review_app/users.json` with the six
reviewer accounts below. All start with PIN `1234`. PINs are stored as
PBKDF2-SHA256 hashes (100k iterations) using the same pattern as
`auth/pin_auth.py`.

| user_id  | Name                 | Site             | Default PIN |
|----------|----------------------|------------------|:-----------:|
| doc_01   | Dr. Naveen Salins    | KMC Manipal      | 1234        |
| doc_02   | Dr. Jenifer Jeba     | CMC Vellore      | 1234        |
| doc_03   | Dr. Arun Ghoshal     | KMC Manipal      | 1234        |
| doc_04   | Dr. Sreedevi Warrier | CCF Coimbatore   | 1234        |
| doc_05   | Dr. Ravi Kannan      | CCHRC Silchar    | 1234        |
| doc_06   | Dr. Parth Sharma     | CCHRC Silchar    | 1234        |

Communicate each reviewer's PIN separately. To rotate a PIN, delete
`review_app/users.json` and edit the `SEED_USERS` list in `main.py`.

## Data layout

The app reads from and writes to `data/evaluation/` at the repo root:

```
data/evaluation/
├── vignettes/{vignette_id}.json    # 40 vignettes (read)
├── rubric.json                      # rubric definition (read; falls back to bundled default)
├── rag_outputs/{vignette_id}.json   # RAG-pipeline output for each vignette (read)
└── expert_reviews/{review_id}.json  # one submitted review per file (write)
```

Saved review JSON shape:

```json
{
  "review_id": "uuid",
  "reviewer_id": "doc_01",
  "reviewer_name": "Dr. Naveen Salins",
  "vignette_id": "ONC-PAIN-DIFFICULT-001",
  "submitted_at": "2026-05-14T14:30:00Z",
  "dimension_scores": {"clinical_accuracy": 4, "safety": 5, ...},
  "sub_item_scores": {"CA-1": true, "CA-2": true, ..., "IC-5": false},
  "overall_score": 8,
  "comments": "...",
  "corrections": "...",
  "tags": ["clinically_strong", "india_context_partial"],
  "would_recommend_for_clinical_use": "yes",
  "flag_for_committee_review": false,
  "time_to_review_seconds": 540
}
```

Only one review per `(reviewer_id, vignette_id)` pair is kept; re-submitting
overwrites in place.

## Routes

| Route                         | Purpose                                              |
|-------------------------------|------------------------------------------------------|
| `GET /`                       | Redirects to `/login` or `/dashboard`                |
| `GET /login`                  | PIN login form                                       |
| `POST /login`                 | Verifies PIN, sets signed session cookie             |
| `GET /logout`                 | Clears the session cookie                            |
| `GET /dashboard?filter=…`     | All 40 vignettes; filter All / Pending / My Reviews  |
| `GET /review/{vignette_id}`   | Three-column review screen                           |
| `POST /review/{vignette_id}`  | Submit (or update) a review                          |
| `GET /progress`               | 6 × 40 progress matrix                               |
| `GET /export.csv`             | All reviews flattened to CSV                         |
| `GET /healthz`                | Liveness probe                                       |

## Session security

Sessions are signed cookies (HMAC-SHA256) with an 8-hour TTL. The signing
secret is auto-generated to `review_app/.session_secret` on first launch (mode
0600). Delete that file to invalidate every active session.

## Dependencies

Already installed in the repo venv: `fastapi`, `starlette`, `uvicorn`,
`jinja2`. No additional packages required.
