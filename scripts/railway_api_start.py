#!/usr/bin/env python3
"""
Start the study server on Railway (the API service).
===================================================
The container's own disk is replaced on every deploy. The server keeps
everything it writes as files under ``data/``, ``uploads/``, ``cache/`` and
``logs/``: patient records, reminders, the study log, the vector index and
uploaded documents. This script puts those four directories on the Railway
volume, then starts the server in one process on ``$PORT``.

What it does, in order:

1. If a volume is mounted (Railway sets ``RAILWAY_VOLUME_MOUNT_PATH``), each of
   the four directories becomes a link to a directory on the volume.

   * First boot: the volume directory is created from the copy the image
     ships (``/opt/palli/seed``), so the bundled vector index is there.
   * Every boot: files the image ships are copied over the volume copy, so a
     new deploy brings the new release record, dose evidence and settings.
     Two exceptions are only seeded once, because the server changes them:
     the vector index (``data/chroma_db``) and its document list
     (``data/document_metadata.json``). Files the server created are never
     touched.

   With no volume the server still starts, on the container disk, and a
   warning says that everything is lost on the next deploy.

2. The Hugging Face cache goes on the volume (``HF_HOME``), so the 2.3 GB
   bge-m3 model is downloaded once, not on every deploy.

3. ``GOOGLE_APPLICATION_CREDENTIALS_JSON``, if set, is written to a private
   file and ``GOOGLE_APPLICATION_CREDENTIALS`` points at it. Railway variables
   hold text, and Vertex AI wants a key file.

4. The server starts with one worker: several workers still keep separate
   state (docs/decisions/latency.md, ticket 18).

``docs/deploy_railway.md`` is the operator guide.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import stat
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

logger = logging.getLogger("railway_api_start")

APP_DIR = Path(os.environ.get("PALLI_APP_DIR", "/app"))
SEED_DIR = Path(os.environ.get("PALLI_SEED_DIR", "/opt/palli/seed"))

# Directories the server writes to, relative to the app directory.
PERSISTENT_DIRS = ("data", "uploads", "cache", "logs")

# Shipped paths the server changes after the first boot. A deploy must not
# replace them, or every redeploy would throw away indexed uploads.
SEED_ONCE = ("data/chroma_db", "data/document_metadata.json")

# Without these the Android app cannot sign in, or no answer can be drafted.
REQUIRED_ANY: Dict[str, Iterable[str]] = {
    "mobile sign-in": ("MOBILE_JWT_SECRET",),
    "Gemini access": ("GEMINI_API_KEY", "GOOGLE_CLOUD_PROJECT"),
}
# Without these a part of the server is off, but the Android app works.
RECOMMENDED = ("ADMIN_JWT_SECRET", "SARVAM_API_KEY", "STUDY_LOG_SALT")


def _seed_once_root(rel: Path) -> Optional[str]:
    """The seed-once path that `rel` is, or is under; None if it is neither."""
    text = rel.as_posix()
    for p in SEED_ONCE:
        if text == p or text.startswith(p + "/"):
            return p
    return None


def _copy_entry(src: Path, dst: Path) -> None:
    """Copy one file or link, replacing what is at `dst`."""
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.is_dir():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_symlink():
        os.symlink(os.readlink(src), dst)
    else:
        shutil.copy2(src, dst)


def refresh_from_seed(seed: Path, target: Path, top: str) -> List[str]:
    """
    Copy every file under `seed` into `target`. A seed-once path is skipped
    as a whole once it exists in `target`: the server owns it from then on,
    and putting back a file it deleted would corrupt the index. Returns the
    relative paths that were written.

    `top` is the directory name (such as "data") that seed-once paths are
    relative to.
    """
    # Decided before copying, so a first boot copies the whole seed-once tree.
    owned = {p for p in SEED_ONCE
             if p.startswith(top + "/") and (target / Path(p).relative_to(top)).exists()}
    written: List[str] = []
    if not seed.exists():
        return written
    for root, dirs, files in os.walk(seed, followlinks=False):
        root_path = Path(root)
        # Links to directories are listed in `dirs` but not walked; copy them
        # as links.
        names = list(files) + [d for d in dirs if (root_path / d).is_symlink()]
        for name in names:
            src = root_path / name
            rel = src.relative_to(seed)
            dst = target / rel
            if _seed_once_root(Path(top) / rel) in owned:
                continue
            _copy_entry(src, dst)
            written.append((Path(top) / rel).as_posix())
    return written


def link_to_volume(app_dir: Path, seed_dir: Path, volume: Path) -> Dict[str, int]:
    """
    Make each persistent directory under `app_dir` a link to the same name on
    `volume`, filling the volume copy from `seed_dir`. Safe to run on every
    boot. Returns, for each directory, how many shipped files were written.
    """
    counts: Dict[str, int] = {}
    for name in PERSISTENT_DIRS:
        on_volume = volume / name
        on_volume.mkdir(parents=True, exist_ok=True)
        counts[name] = len(refresh_from_seed(seed_dir / name, on_volume, name))

        in_app = app_dir / name
        if in_app.is_symlink():
            if Path(os.readlink(in_app)) == on_volume:
                continue
            in_app.unlink()
        elif in_app.is_dir():
            shutil.rmtree(in_app)
        elif in_app.exists():
            in_app.unlink()
        in_app.symlink_to(on_volume, target_is_directory=True)
    return counts


def fill_without_volume(app_dir: Path, seed_dir: Path) -> None:
    """No volume: make sure the persistent directories exist on the container disk."""
    for name in PERSISTENT_DIRS:
        in_app = app_dir / name
        if in_app.is_symlink() and not in_app.exists():
            in_app.unlink()
        if not in_app.exists():
            refresh_from_seed(seed_dir / name, in_app, name)
            in_app.mkdir(parents=True, exist_ok=True)


def write_google_credentials(env: MutableMapping[str, str], path: Path) -> Optional[Path]:
    """Write GOOGLE_APPLICATION_CREDENTIALS_JSON to `path` (mode 600) and point the SDK at it."""
    raw = env.get("GOOGLE_APPLICATION_CREDENTIALS_JSON", "").strip()
    if not raw:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w") as f:
        f.write(raw)
    env["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)
    return path


def missing_settings(env: Mapping[str, str]) -> List[str]:
    """Each required purpose that has none of its variables set."""
    return [
        f"{purpose} (set {' or '.join(names)})"
        for purpose, names in REQUIRED_ANY.items()
        if not any(env.get(n, "").strip() for n in names)
    ]


def server_command(env: Mapping[str, str]) -> List[str]:
    """The server's command line. PALLI_SERVER_EXTRA_ARGS adds flags such as --rag-method."""
    port = env.get("PORT", "8000")
    cmd = [
        sys.executable, "simple_rag_server.py",
        "--host", "0.0.0.0",
        "--port", port,
        "--no-ngrok",
        "--workers", "1",
    ]
    cmd += shlex.split(env.get("PALLI_SERVER_EXTRA_ARGS", ""))
    return cmd


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s railway-start %(levelname)s %(message)s")
    env = os.environ

    missing = missing_settings(env)
    if missing:
        logger.error("Not starting. Missing Railway variables: %s. See docs/deploy_railway.md.",
                     "; ".join(missing))
        sys.exit(2)
    for name in RECOMMENDED:
        if not env.get(name, "").strip():
            logger.warning("%s is not set; the part of the server that needs it is off.", name)
    if not env.get("WEB_VOICE_ACCESS_CODE", "").strip():
        logger.warning("WEB_VOICE_ACCESS_CODE is not set: anyone with the address can talk to "
                       "the web voice demo at /voice/ (docs/deploy_voice_demo.md).")

    volume_path = env.get("RAILWAY_VOLUME_MOUNT_PATH", "").strip()
    if volume_path:
        volume = Path(volume_path)
        counts = link_to_volume(APP_DIR, SEED_DIR, volume)
        logger.info("Data directories on volume %s; shipped files written: %s", volume, counts)
        env.setdefault("HF_HOME", str(volume / "hf"))
    else:
        logger.warning(
            "NO VOLUME IS MOUNTED. Records, reminders, the study log and uploads are on the "
            "container disk and are deleted by the next deploy. Attach a volume (docs/deploy_railway.md)."
        )
        fill_without_volume(APP_DIR, SEED_DIR)

    if write_google_credentials(env, Path("/tmp/palli/gcp-key.json")):
        logger.info("Vertex AI key file written from GOOGLE_APPLICATION_CREDENTIALS_JSON")

    cmd = server_command(env)
    logger.info("Starting: %s", shlex.join(cmd))
    os.chdir(APP_DIR)
    os.execv(cmd[0], cmd)


if __name__ == "__main__":
    main()
