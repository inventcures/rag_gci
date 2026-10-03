#!/usr/bin/env python3
"""
Export the mobile API's OpenAPI schema
======================================
The Android client is generated from this schema rather than written by hand, and
`:core:api:verifyApiContract` compares the live schema against the committed
snapshot.

Why this matters. A hand-written client drifts silently. The app compiles, a field
is renamed on the server, and requests succeed while answers arrive empty — which
in this product means an ASHA worker is told nothing and does not know why. That
failure survives code review and only appears in the field.

Usage:
    python3 scripts/export_openapi_contract.py           # print schema to stdout
    python3 scripts/export_openapi_contract.py --write   # update the committed snapshot
    python3 scripts/export_openapi_contract.py --check   # verify, exit 1 on drift

The comparison lives here rather than in the Gradle task because this module
already owns the schema and Python has a JSON parser available; the Gradle
scripting classpath does not. The build task shells out to --check.

`--check` compares structurally rather than textually. The server emits key
orderings and metadata that change without the contract changing, so a raw text
diff would report drift on every run and train everyone to ignore the check.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = ROOT / "android-app" / "core" / "api" / "contract" / "openapi.json"


def build_schema() -> dict:
    """Build the FastAPI app exactly as the server does, then read its schema."""
    sys.path.insert(0, str(ROOT))

    from fastapi import FastAPI

    from mobile_api.router import mobile_router

    app = FastAPI(
        title="Palli Sahayak Mobile API",
        version="1",
        description=(
            "Mobile API for the Palli Sahayak Android client. The Kotlin client is "
            "generated from this schema; the committed copy is the contract."
        ),
    )
    # mobile_router already declares prefix="/api/mobile/v1". Adding it again here
    # produced /api/mobile/v1/api/mobile/v1/... and would have generated a client
    # whose every path 404s.
    app.include_router(mobile_router)
    return app.openapi()


def _resolve(schema: dict, root: dict, depth: int = 0) -> dict:
    """
    Follow a local $ref into components/schemas.

    Every response and most request bodies are expressed as a $ref to a named
    component rather than an inline schema. Without resolving those, a renamed
    field inside MobileQueryResponse would change nothing this script can see,
    and the contract check would pass while the client silently stopped binding
    it. That is precisely the failure the check exists to catch.
    """
    if not isinstance(schema, dict) or depth > 10:
        return {}
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema

    node = root
    for part in ref[2:].split("/"):
        if not isinstance(node, dict) or part not in node:
            return {}
        node = node[part]
    if not isinstance(node, dict):
        return {}
    return _resolve(node, root, depth + 1)


def summarise(schema: dict) -> set:
    """
    Reduce a schema to the facts a client actually depends on.

    Compares paths, methods, parameters, and request/response field names and
    types. Follows $ref into components so component-level renames are caught.
    Ignores descriptions, ordering and metadata, which change without the contract
    changing.
    """
    lines = set()
    for path in sorted(schema.get("paths", {})):
        operations = schema["paths"][path]
        if not isinstance(operations, dict):
            continue
        for method in sorted(operations):
            operation = operations[method]
            if not isinstance(operation, dict):
                continue
            lines.add(f"{method.upper()} {path}")

            parameters = operation.get("parameters") or []
            for parameter in sorted(
                parameters,
                key=lambda p: p.get("name", "") if isinstance(p, dict) else "",
            ):
                if not isinstance(parameter, dict):
                    continue
                lines.add(
                    f"  {method.upper()} {path} param {parameter.get('name')}"
                    f" in={parameter.get('in')} required={parameter.get('required')}"
                )

            body = _json_schema(operation.get("requestBody"), schema)
            if body:
                lines.add(f"  {method.upper()} {path} body {_fields(body, schema)}")

            for code, response in sorted((operation.get("responses") or {}).items()):
                lines.add(f"  {method.upper()} {path} response {code}")
                response_schema = _json_schema(response, schema)
                if response_schema:
                    lines.add(
                        f"    {method.upper()} {path} response {code} "
                        f"{_fields(response_schema, schema)}"
                    )
    return lines


def _json_schema(container: Any, root: dict) -> dict:
    """Pull the application/json schema out of a request body or response."""
    if not isinstance(container, dict):
        return {}
    content = container.get("content") or {}
    media = content.get("application/json") or {}
    schema = media.get("schema") or {}
    return _resolve(schema, root) if isinstance(schema, dict) else {}


def _fields(schema: dict, root: dict, depth: int = 0) -> str:
    """
    Field names and types, one level deep.

    One level is deliberate: it is where a rename actually breaks a generated
    client, and recursing would report every nested change without saying which
    one the client would stop binding.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return str(schema.get("type", "any"))
    parts = []
    for name in sorted(properties):
        prop = properties[name] or {}
        if not isinstance(prop, dict):
            parts.append(f"{name}:?")
            continue
        kind = prop.get("type")
        if kind is None:
            resolved = _resolve(prop, root, depth + 1) if depth < 3 else {}
            kind = prop.get("$ref") or resolved.get("type") or "?"
        if kind == "array":
            items = prop.get("items") or {}
            inner = _resolve(items, root, depth + 1) if isinstance(items, dict) else {}
            kind = f"array[{inner.get('type') or items.get('$ref') or '?'}]"
        parts.append(f"{name}:{kind}")
    return ",".join(parts)


def check_drift() -> int:
    """Exit 1 and describe the drift when the committed contract is stale."""
    if not CONTRACT_PATH.exists():
        print(
            "API CONTRACT MISSING: "
            f"{CONTRACT_PATH.relative_to(ROOT)} does not exist.\n"
            "Create it with: python3 scripts/export_openapi_contract.py --write",
            file=sys.stderr,
        )
        return 1

    committed = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    fresh = build_schema()

    before, after = summarise(committed), summarise(fresh)
    if before == after:
        print("API contract matches the server schema.")
        return 0

    removed = sorted(before - after)
    added = sorted(after - before)

    print("API CONTRACT DRIFT", file=sys.stderr)
    print("", file=sys.stderr)
    print("The server schema has changed since the committed contract in", file=sys.stderr)
    print(f"{CONTRACT_PATH.relative_to(ROOT)}.", file=sys.stderr)
    print("", file=sys.stderr)
    for line in removed:
        print(f"  - {line}", file=sys.stderr)
    for line in added:
        print(f"  + {line}", file=sys.stderr)
    if len(removed) + len(added) > 60:
        print("  ... and more", file=sys.stderr)
    print("", file=sys.stderr)
    print("If this change is intended:", file=sys.stderr)
    print("  python3 scripts/export_openapi_contract.py --write", file=sys.stderr)
    print("then regenerate the Kotlin client and review the diff.", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--write", action="store_true",
        help="Update the committed contract snapshot.",
    )
    mode.add_argument(
        "--check", action="store_true",
        help="Compare against the committed snapshot; exit 1 on drift.",
    )
    args = parser.parse_args()

    if args.check:
        return check_drift()

    try:
        schema = build_schema()
    except Exception as exc:  # noqa: BLE001
        # A hard failure here must be loud: the Gradle check treats empty output
        # as drift, which would report the wrong cause.
        print(f"Failed to build OpenAPI schema: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False)

    if args.write:
        CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONTRACT_PATH.write_text(rendered + "\n", encoding="utf-8")
        print(
            f"Wrote {CONTRACT_PATH.relative_to(ROOT)} "
            f"({len(schema.get('paths', {}))} paths)",
            file=sys.stderr,
        )
        return 0

    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
