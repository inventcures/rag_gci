#!/usr/bin/env python3
"""
Mint an admin token for a named person (auth/policy.py).

Admin, study, clinical-data and upload routes need it, and so do the Gradio
admin panel and the study dashboard. The name becomes the token's subject, which
the voice-provider override records as the actor. Needs ADMIN_JWT_SECRET (at
least 32 characters) in the environment, the same value the server runs with.

    python scripts/issue_admin_token.py dr-ashish            # valid 12 hours
    python scripts/issue_admin_token.py dr-ashish --hours 2

Send it as `Authorization: Bearer <token>`, or in a browser as the password of
the sign-in prompt (any user name).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

from auth.policy import issue_admin_token  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Print a fresh admin token for the named person."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("subject", help="the administrator's name, recorded as the audit actor")
    parser.add_argument("--hours", type=float, default=12.0, help="how long the token is valid (default 12)")
    args = parser.parse_args(argv)
    load_dotenv()
    try:
        token = issue_admin_token(args.subject, hours=args.hours)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    sys.stdout.write(token + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
