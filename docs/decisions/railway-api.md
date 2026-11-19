# Railway deployment of the study server: decision log

Asked for on 2026-11-19. There is no domain yet, but there is a Railway account. The Android test APK needs a server that a real phone can reach. Before this change, `railway.toml` and `nixpacks.toml` deployed only the expert review console (`review_app/`), not the study server.

Operator guide: [docs/deploy_railway.md](../deploy_railway.md).

## Decisions

| # | Decision | Why |
|---|---|---|
| RW1 | This deployment is for testing with made-up people only. The guide, the PR and `.railway/railway.ts` say so. | Railway lists four regions: two in the USA, Amsterdam and Singapore ([regions](https://docs.railway.com/deployments/regions), checked 2026-11-19). None is in India, and identifiable study data must stay India-hosted. |
| RW2 | Region Singapore (`asia-southeast1-eqsg3a`). | Nearest to India, so the lowest network latency for phones there. |
| RW3 | Declare the API service in Railway Infrastructure as Code (`.railway/railway.ts`), not in a second `railway.toml`. The file is a partial (`export const partial = "api"`) that declares only the `api` service and its volume. | Railway deprecated `railway.toml` and `railway.json`: new services cannot use them, and existing ones stop being read on 2026-12-01 ([guide](https://railway.com/guides/railway-infrastructure-as-code)). A partial manages only what it declares, so an apply cannot remove the review console. The guide also gives the same settings as dashboard steps. |
| RW4 | Pin the IaC SDK at `railway@3.16.0` in `.railway/package.json`. | The SDK is in beta and says it will break. 3.16.0 is the newest release older than two weeks. The file was evaluated and type-checked against it. It was not planned against a live project, because this session has no Railway account. |
| RW5 | Build from `Dockerfile.api` (Docker builder), not Nixpacks or Railpack. | `nixpacks.toml` at the root is the review console's and installs only its requirements. A Dockerfile is explicit, can install CPU-only torch and the system tools, and builds the same image locally. |
| RW6 | The file is `Dockerfile.api`, not `Dockerfile`. A test fails if a root `Dockerfile` appears. | Railway auto-detects a root `Dockerfile` for a service whose builder is not set. When the review console stops reading `railway.toml` on 2026-12-01, a root `Dockerfile` could make it build the study server. |
| RW7 | Install torch from the PyTorch CPU index before `requirements.txt`. | The default Linux wheel pulls several GB of CUDA libraries that a Railway container cannot use. It makes the image and every build much larger for nothing. |
| RW8 | One volume at `/data`. The start script makes `data/`, `uploads/`, `cache/` and `logs/` links into it. | Railway allows one volume per service. The server writes to all four as relative paths, and many modules build their own paths under `data/`, so a link is the one change that covers all of them without touching server code. |
| RW9 | Shipped files are copied over the volume copy on every boot, except `data/chroma_db/` and `data/document_metadata.json`, which are copied once. | `data/study/release.yaml`, `dosage_evidence.json` and settings belong to the release, so a deploy must bring the new ones. The index and its document list are changed by uploads; replacing them on deploy would lose indexed documents, and restoring a segment the server deleted would corrupt the index. Runtime files such as `data/study/logs/` are never in the image, so they are never touched. |
| RW10 | Put the Hugging Face cache on the volume (`HF_HOME=/data/hf`), rather than baking bge-m3 into the image. | The model is downloaded once and kept across deploys. Baking it in adds 2.3 GB to every build and push. The cost is a slow first boot, so the health check timeout is 900 seconds. |
| RW11 | Volume size 5 GB. | The Hobby plan maximum. bge-m3 is 2.3 GB, the shipped index is 11 MB, and test data is small. Growing it later is safe; shrinking it is destructive. |
| RW12 | One replica, one worker (`--workers 1`), set in the start script and the IaC file. | Railway does not allow replicas with a volume. Several workers still keep per-worker state (docs/decisions/latency.md, ticket 18). |
| RW13 | The start script refuses to start without `MOBILE_JWT_SECRET` or a Gemini credential. It only warns for `ADMIN_JWT_SECRET`, `SARVAM_API_KEY` and `STUDY_LOG_SALT`. | Without the mobile secret the server starts but the whole mobile API is absent, and the phone sees 404s with no explanation. Without a Gemini credential no answer can be drafted. The others switch off one part, and the server is still useful. |
| RW14 | With no volume the server still starts, with a loud warning. | It allows a first look before the volume is attached. All data in this deployment is made up, so losing it is an inconvenience. The warning says it in capitals. |
| RW15 | Secrets are `preserve()` in the IaC file and are set in the dashboard. No secret is generated in the file. | The repository is public. The SDK's `ctx.randomString` is a hash of a fixed label, so anyone can compute it; it is not a secret. |
| RW16 | Support `GOOGLE_APPLICATION_CREDENTIALS_JSON`: the start script writes it to a mode 600 file and sets `GOOGLE_APPLICATION_CREDENTIALS`. | Vertex AI keeps model calls in `asia-south1` and needs a key file. Railway variables hold text only. The Gemini API key works too and is simpler. |
| RW17 | `PORT=8000` is set in the IaC file, and the guide says to generate the domain for port 8000. | A fixed port makes the domain setting and the health check predictable. |
| RW18 | Add a `.dockerignore` for `.git`, `.env`, the Android project and large media. | Keeps secrets from a local `.env` out of a locally built image and keeps the build context small. It excludes nothing the review console's build reads. |

## Verified here

* `tests/test_railway_api_start.py`: first boot, redeploy (new release record arrives; index, records, logs and uploads stay), a second run in one container, no volume, the variable check, the key file mode, and the server command.
* `.railway/railway.ts` evaluates with the pinned SDK and type-checks under `tsc --strict`.
* The image was built and booted locally, except the `apt-get` step and the PyTorch CPU index, which this sandbox's network blocks (both were replaced by PyPI torch and no system packages for the local build only). See the PR for the boot result.

## Not verified, and why

* `railway config plan` and `apply` against a real project: no Railway account in this session.
* The real Railway build, the first boot model download, and the phone connecting: they need the service to exist.

## For the operator

| # | Item |
|---|---|
| O1 | The review console still deploys from `railway.toml`, which Railway stops reading on 2026-12-01. After that its build and start settings must come from the dashboard or from an IaC partial of its own. Recommendation: migrate it before that date, in the dashboard (Settings of the service: move the values in, then clear the config file path), and then delete `railway.toml` and `nixpacks.toml` in a follow-up PR. |
| O2 | The study deployment needs an India-hosted server (for example a VM in `asia-south1`). This Railway service is not that. |
