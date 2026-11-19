# Deploy the study server on Railway (testing only)

This guide puts the study server (`simple_rag_server.py`) on Railway, so that the Android app on a real phone can talk to it. It gives a public `https://<name>.up.railway.app` address. No domain is needed.

> **Testing with made-up people only.** Railway has no region in India. The nearest is Singapore (`asia-southeast1-eqsg3a`, see [Railway regions](https://docs.railway.com/deployments/regions)). The study rule is that identifiable data stays in India. Do not enter a real participant, caregiver or patient into this deployment. The study deployment needs an India-hosted server.

The expert review console is a different Railway service. It still deploys from `railway.toml` and `nixpacks.toml`. This guide does not change it.

## What is in the repository

| File | What it does |
|---|---|
| `Dockerfile.api` | The image: Python 3.12, CPU-only torch, `requirements.txt`, system tools for uploads and voice notes. |
| `scripts/railway_api_start.py` | The start command. It puts the server's data on the volume, checks the required variables, and starts one server process on port 8000. |
| `.railway/railway.ts` | The Railway service, its volume, its region and its health check, as Railway Infrastructure as Code. |
| `.dockerignore` | Keeps `.git`, `.env` and the Android project out of the image. |

## What the server needs

| Need | Value | Why |
|---|---|---|
| Memory | 4 GB at least, 8 GB is safe | bge-m3 uses about 2 to 3 GB. Railway bills the memory that is used, not a fixed size. |
| Volume | 5 GB, mounted at `/data` | Records, reminders, the study log, uploads, the vector index and the model cache are files. The container disk is replaced on every deploy. |
| Processes | 1 | Several server processes still keep separate state. A volume also cannot be shared by replicas. |
| Plan | Hobby or Pro | The free plan allows a 0.5 GB volume, which is too small for the model cache. |

The first boot downloads bge-m3 (2.3 GB) to the volume before the server listens. It can take 5 to 10 minutes. Later boots read it from the volume. The health check waits up to 15 minutes.

## Steps only you can do

The repository cannot act in your Railway account. Do these steps in order.

### 1. Make the secrets

Run this on your computer once for each secret, and keep the output:

```bash
openssl rand -hex 32
```

You need two: `MOBILE_JWT_SECRET` and `ADMIN_JWT_SECRET`. They must be different.

### 2. Create the service

Use the command line (recommended) or the dashboard.

**Command line.** Install the Railway CLI, version 5.42.1 or newer. Then, from the repository root:

```bash
railway login
railway link                 # choose the project, or create one with: railway init
cd .railway && npm ci && cd ..
railway config plan          # read it: it must only add "api" and "api-data"
railway config apply
```

**Dashboard.** If you prefer clicking:

1. New Project, or open the existing one. Click **New**, then **GitHub Repo**, then `inventcures/rag_gci`. Name the service `api`.
2. Service **Settings**, **Build**: builder **Dockerfile**, path `Dockerfile.api`.
3. **Deploy**: start command `python scripts/railway_api_start.py`, health check path `/health`, health check timeout `900`.
4. **Region**: Southeast Asia (Singapore). **Replicas**: 1.
5. If the service settings say they come from `railway.toml`, clear the config file path. That file is the review console's.
6. Right-click the service, **Attach volume**, mount path `/data`.

### 3. Set the variables

Service **Variables**. Railway sets `RAILWAY_VOLUME_MOUNT_PATH` itself when the volume is attached.

| Variable | Required | Value |
|---|---|---|
| `PORT` | yes | `8000` |
| `MOBILE_JWT_SECRET` | yes | The first secret from step 1. Without it the Android API is off, so the server does not start. |
| `GEMINI_API_KEY` | yes, or Vertex | A Gemini API key. |
| `ADMIN_JWT_SECRET` | recommended | The second secret. Without it the admin panel and admin routes refuse every request. |
| `SARVAM_API_KEY` | for voice | Sarvam speech to text and text to speech. |
| `STUDY_LOG_SALT` | recommended | Any long random string. It salts the hashed IDs in the study log. |
| `CORS_ALLOWED_ORIGINS` | no | Only for a browser on another origin. The Android app does not need it. |
| `PALLI_SERVER_EXTRA_ARGS` | no | Extra server flags, for example `--rag-method kg`. |

**Vertex AI in place of the Gemini API key.** Set `GOOGLE_CLOUD_PROJECT` and put the whole service account key JSON in `GOOGLE_APPLICATION_CREDENTIALS_JSON`. The start script writes it to a private file. Vertex keeps the model calls in `asia-south1`. The Gemini API key path does not pin a region.

### 4. Get the address

Service **Settings**, **Networking**, **Generate Domain**, port `8000`. You get `https://<name>.up.railway.app`.

Check it:

```bash
curl https://<name>.up.railway.app/health
```

`"status": "healthy"` means the vector index loaded. `"degraded"` means the server is up but cannot answer; the deploy logs say why.

### 5. Point the test APK at it

In GitHub, **Settings**, **Secrets and variables**, **Actions**, **Variables**, set `PALLI_TEST_BASE_URL` to the address from step 4, with a trailing slash: `https://<name>.up.railway.app/`. The next test APK build uses it.

### 6. Admin sign-in (optional)

Railway's service shell has the variables set, so you can mint a token there:

```bash
railway ssh -s api -- python scripts/issue_admin_token.py <your-name>
```

## How data survives a deploy

The start script makes `data/`, `uploads/`, `cache/` and `logs/` links to the same names on the volume.

* First boot: each volume directory is filled from the copy in the image, so the bundled vector index is there.
* Every boot: files that the image ships are copied over the volume copy. A deploy therefore brings the new `data/study/release.yaml`, dose evidence and settings.
* Two shipped paths are copied only once, because the server changes them: `data/chroma_db/` and `data/document_metadata.json`.
* Files the server creates (records, reminders, study logs, uploads) are never touched.

With no volume the server still starts, and the log says in capitals that all data goes with the next deploy.

A redeploy has a short outage. Railway cannot run two deployments that mount the same volume.

## Problems

| Symptom | Cause and fix |
|---|---|
| Deploy log: `Not starting. Missing Railway variables` | Set the variables it names (step 3). |
| Build fails: `Build context is missing the server files` | Something removed the server from the build. Check that the service builds with `Dockerfile.api` and not with the review console's settings. |
| The service runs the review console | It read `railway.toml`. Clear the config file path (step 2, dashboard item 5) and set the builder and start command again. |
| Health check times out on the first deploy | The model download was slow. Redeploy; the part already downloaded stays on the volume. |
| The app gets 404 on `/api/mobile/v1/...` | `MOBILE_JWT_SECRET` is empty, or `mobile.enabled` is false in `config.yaml`. |
| Out of memory | Raise the memory limit in the service settings to 8 GB. |

## Later

When the domain is ready, add `api.pallisahayak.org` as a custom domain on this service (Settings, Networking) and set `PALLI_TEST_BASE_URL` to `https://api.pallisahayak.org/`. The study itself needs an India-hosted server; this one stays for testing.

To demo the web voice page at an event, see [deploy_voice_demo.md](deploy_voice_demo.md).

Decisions behind this setup: [docs/decisions/railway-api.md](decisions/railway-api.md).
