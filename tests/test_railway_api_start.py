"""The Railway start script keeps server data on the volume across deploys."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("railway_api_start", ROOT / "scripts" / "railway_api_start.py")
start = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(start)


def _seed(tmp_path: Path) -> Path:
    """An image copy shaped like the real one: shipped files, the index, a link."""
    seed = tmp_path / "seed"
    (seed / "data" / "study").mkdir(parents=True)
    (seed / "data" / "study" / "release.yaml").write_text("release: 1\n")
    (seed / "data" / "chroma_db" / "seg-a").mkdir(parents=True)
    (seed / "data" / "chroma_db" / "chroma.sqlite3").write_text("shipped index")
    (seed / "data" / "chroma_db" / "seg-a" / "header.bin").write_text("a")
    (seed / "data" / "document_metadata.json").write_text("{}")
    (seed / "data" / "graphrag").mkdir()
    os.symlink("../../uploads", seed / "data" / "graphrag" / "input")
    for name in ("uploads", "cache", "logs"):
        (seed / name).mkdir()
    return seed


def _app(tmp_path: Path, seed: Path) -> Path:
    """The app directory as the image ships it: a real copy of each directory."""
    app = tmp_path / "app"
    app.mkdir()
    for name in start.PERSISTENT_DIRS:
        (app / name).mkdir()
    (app / "data" / "from-image.txt").write_text("x")
    return app


def test_first_boot_links_every_directory_and_seeds_the_index(tmp_path):
    seed = _seed(tmp_path)
    app = _app(tmp_path, seed)
    volume = tmp_path / "volume"

    start.link_to_volume(app, seed, volume)

    for name in start.PERSISTENT_DIRS:
        assert (app / name).is_symlink()
        assert (app / name).resolve() == (volume / name).resolve()
    assert (app / "data" / "chroma_db" / "chroma.sqlite3").read_text() == "shipped index"
    assert (app / "data" / "study" / "release.yaml").read_text() == "release: 1\n"
    # The GraphRAG input link still points at uploads, now on the volume.
    assert os.readlink(volume / "data" / "graphrag" / "input") == "../../uploads"
    assert (volume / "data" / "graphrag" / "input").resolve() == (volume / "uploads").resolve()


def test_redeploy_keeps_server_data_and_brings_new_shipped_files(tmp_path):
    seed = _seed(tmp_path)
    app = _app(tmp_path, seed)
    volume = tmp_path / "volume"
    start.link_to_volume(app, seed, volume)

    # The server indexes an upload (changing the index, dropping a segment)
    # and writes records and a study log.
    (volume / "data" / "chroma_db" / "chroma.sqlite3").write_text("indexed upload")
    (volume / "data" / "chroma_db" / "seg-a" / "header.bin").unlink()
    (volume / "data" / "chroma_db" / "seg-a").rmdir()
    (volume / "data" / "document_metadata.json").write_text('{"doc": 1}')
    (volume / "data" / "study" / "logs").mkdir()
    (volume / "data" / "study" / "logs" / "2026-11-19.jsonl").write_text("entry\n")
    (volume / "uploads" / "guide.pdf").write_text("pdf")

    # A new deploy: a fresh app directory and a new release record.
    (seed / "data" / "study" / "release.yaml").write_text("release: 2\n")
    new_app = tmp_path / "app2"
    new_app.mkdir()
    for name in start.PERSISTENT_DIRS:
        (new_app / name).mkdir()
    start.link_to_volume(new_app, seed, volume)

    assert (new_app / "data" / "study" / "release.yaml").read_text() == "release: 2\n"
    assert (new_app / "data" / "chroma_db" / "chroma.sqlite3").read_text() == "indexed upload"
    assert not (volume / "data" / "chroma_db" / "seg-a").exists()
    assert (new_app / "data" / "document_metadata.json").read_text() == '{"doc": 1}'
    assert (new_app / "data" / "study" / "logs" / "2026-11-19.jsonl").read_text() == "entry\n"
    assert (new_app / "uploads" / "guide.pdf").read_text() == "pdf"


def test_running_twice_in_one_container_is_harmless(tmp_path):
    seed = _seed(tmp_path)
    app = _app(tmp_path, seed)
    volume = tmp_path / "volume"
    start.link_to_volume(app, seed, volume)
    (volume / "data" / "records.json").write_text("kept")

    start.link_to_volume(app, seed, volume)

    assert (app / "data").is_symlink()
    assert (app / "data" / "records.json").read_text() == "kept"


def test_without_a_volume_the_directories_exist_on_the_container_disk(tmp_path):
    seed = _seed(tmp_path)
    app = tmp_path / "app"
    app.mkdir()

    start.fill_without_volume(app, seed)

    for name in start.PERSISTENT_DIRS:
        assert (app / name).is_dir()
    assert (app / "data" / "study" / "release.yaml").exists()


def test_missing_settings_names_each_purpose():
    assert start.missing_settings({"MOBILE_JWT_SECRET": "s", "GEMINI_API_KEY": "k"}) == []
    assert start.missing_settings({"MOBILE_JWT_SECRET": "s", "GOOGLE_CLOUD_PROJECT": "p"}) == []
    missing = start.missing_settings({"GEMINI_API_KEY": " "})
    assert len(missing) == 2
    assert "MOBILE_JWT_SECRET" in missing[0]


def test_google_key_is_written_privately(tmp_path):
    env = {"GOOGLE_APPLICATION_CREDENTIALS_JSON": '{"type": "service_account"}'}
    path = start.write_google_credentials(env, tmp_path / "key.json")
    assert path is not None
    assert env["GOOGLE_APPLICATION_CREDENTIALS"] == str(path)
    assert (path.stat().st_mode & 0o777) == 0o600
    assert start.write_google_credentials({}, tmp_path / "none.json") is None


def test_server_runs_one_worker_on_railways_port():
    cmd = start.server_command({"PORT": "8123", "PALLI_SERVER_EXTRA_ARGS": "--rag-method kg"})
    assert cmd[0] == sys.executable
    assert cmd[cmd.index("--port") + 1] == "8123"
    assert cmd[cmd.index("--workers") + 1] == "1"
    assert "--no-ngrok" in cmd
    assert cmd[-2:] == ["--rag-method", "kg"]


@pytest.mark.parametrize("path", ["Dockerfile.api", ".railway/railway.ts"])
def test_deploy_files_use_the_start_script(path):
    """The image and the Railway service both start through the script."""
    assert "scripts/railway_api_start.py" in (ROOT / path).read_text()


def test_no_root_dockerfile():
    """A root Dockerfile would be picked up by the review console service."""
    assert not (ROOT / "Dockerfile").exists()
