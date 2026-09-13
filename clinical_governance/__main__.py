"""Run the isolated workbench, optionally with synthetic unapproved examples."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import secrets
import tempfile

import uvicorn

from .api import GovernanceRuntime, create_app
from .models import Actor
from .store import GovernanceStore


def seed_demo(store, actor):
    """Only proposals. No fabricated clinician review or validated protocol."""
    valid_until = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    text = (
        "SYNTHETIC SOFTWARE FIXTURE. Not medical guidance.\n"
        "During a synthetic encounter, ask the participant to describe the agreed plan.\n"
        "Footnote: this statement is for interface testing only.\n"
    )
    source = store.execute(
        actor,
        "source",
        {
            "title": "Synthetic communication protocol",
            "version": "demo-1",
            "original": text.encode(),
            "normalized_text": text,
            "coverage_gaps": ["Clinical and language review have not been performed."],
            "rights": {
                "authorized": True,
                "authorization_reference": "Authored synthetic UI fixture",
                "purposes": ["information"],
                "valid_until": valid_until,
            },
        },
    )
    store.execute(
        actor,
        "propose",
        {
            "draft": {
                "statement": "Ask about the agreed plan during a synthetic encounter.",
                "evidence": [
                    {
                        "source_id": source["id"],
                        "normalized_sha256": source["normalized_sha256"],
                        "start": 0,
                        "end": len(text),
                        "excerpt": text,
                    }
                ],
                "condition": {
                    "op": "eq",
                    "field": "synthetic_encounter",
                    "value": True,
                },
                "purposes": ["information"],
                "keywords": ["plan", "योजना"],
                "explanations": {
                    "en": "Synthetic demonstration only: please describe the agreed plan."
                },
                "exceptions": ["Not for patient-care use"],
                "valid_until": valid_until,
            }
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description="Palli Sahayak clinical knowledge workbench"
    )
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Local synthetic proposals with no clinical approval",
    )
    args = parser.parse_args()
    runtime = None
    if args.demo:
        directory = Path(tempfile.mkdtemp(prefix="palli-governance-demo-"))
        token = secrets.token_urlsafe(32)
        token_path = directory / "access-token.txt"
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(token)
        actor = Actor(
            "synthetic-demo-editor",
            "synthetic-demo",
            frozenset({"editor", "operator", "auditor", "requester"}),
        )
        store = GovernanceStore(
            directory / "governance.sqlite3", secrets.token_bytes(32)
        )
        seed_demo(store, actor)
        runtime = GovernanceRuntime(
            store, {hashlib.sha256(token.encode()).hexdigest(): actor}, "parallel"
        )
        print(f"Synthetic-only workbench: http://127.0.0.1:{args.port}/governance")
        print(
            f"Local demo access token is in {token_path}. No clinical approval is seeded."
        )
    uvicorn.run(create_app(runtime), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
