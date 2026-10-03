#!/usr/bin/env python3
"""
Generate the Kotlin API client from the committed OpenAPI contract
================================================================
Reads the contract snapshot and emits Kotlin data classes plus a Retrofit
interface for the operations the Android client uses.

Generation is from the *committed contract*, not the live server, so the emitted
source is reproducible and reviewable, and so a server change shows up as a diff
in this output rather than as a surprise at runtime. `verifyApiContract` gates the
pairing: the contract is the thing that must match the server, and this script
turns it into Kotlin.

Scope is deliberately narrow: data classes for the schemas the client binds, and
suspend functions for the routes it calls. Endpoint parameters are typed where the
schema declares them.

Usage:
    python3 scripts/generate_kotlin_client.py            # write into :core:api
    python3 scripts/generate_kotlin_client.py --check    # fail if output is stale
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "android-app" / "core" / "api" / "contract" / "openapi.json"
OUT_DIR = ROOT / "android-app" / "core" / "api" / "src" / "main" / "java" / "org" / "inventcures" / "pallisahayak" / "api" / "generated"

PACKAGE = "org.inventcures.pallisahayak.api.generated"

# Routes the client calls. Anything outside this list is not generated: an unused
# endpoint is not a dependency the client should carry.
CLIENT_OPERATIONS = [
    ("POST", "/api/mobile/v1/query", "query"),
    ("POST", "/api/mobile/v1/query/voice", "voiceQuery"),
    ("GET", "/api/mobile/v1/cache/bundle", "cacheBundle"),
    ("POST", "/api/mobile/v1/auth/login", "login"),
]

KOTLIN_KEYWORDS = {
    "object", "class", "fun", "val", "var", "return", "when", "is", "in", "as",
    "package", "import", "interface", "internal", "null", "true", "false", "this",
}


def kotlin_type(schema: Dict[str, Any], root: Dict[str, Any], depth: int = 0) -> str:
    """Map a JSON-schema node to a Kotlin type."""
    if not isinstance(schema, dict) or depth > 4:
        return "Any?"

    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
        return ref.rsplit("/", 1)[-1]

    kind = schema.get("type")

    if kind == "array":
        return f"List<{kotlin_type(schema.get('items') or {}, root, depth + 1)}>"
    if kind == "object" or (kind is None and "properties" in schema):
        return "Map<String, Any?>"
    if kind == "string":
        fmt = schema.get("format")
        return "String"
    if kind == "integer":
        return "Long"
    if kind == "number":
        return "Double"
    if kind == "boolean":
        return "Boolean"
    if schema.get("anyOf") or schema.get("oneOf"):
        return "Any?"
    if kind is None:
        return "Any?"
    return "Any?"


def resolve(schema: Dict[str, Any], root: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
    if not isinstance(schema, dict) or depth > 10:
        return {}
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/"):
        node: Any = root
        for part in ref[2:].split("/"):
            if not isinstance(node, dict) or part not in node:
                return {}
            node = node[part]
        return resolve(node, root, depth + 1) if isinstance(node, dict) else {}
    return schema


def name_of(field: str) -> str:
    """A safe Kotlin identifier from a schema property name."""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", field)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"f_{cleaned}"
    if cleaned in KOTLIN_KEYWORDS:
        cleaned = f"{cleaned}_"
    return cleaned


def camel(field: str) -> str:
    parts = re.split(r"[_\-]", field)
    return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])


def emit_data_class(name: str, schema: Dict[str, Any], root: Dict[str, Any]) -> str:
    resolved = resolve(schema, root)
    properties = resolved.get("properties") or {}
    required = set(resolved.get("required") or [])

    header = [
        "/**",
        " * Generated from the API contract. Do not edit by hand; run",
        " * scripts/generate_kotlin_client.py instead.",
        " */",
    ]

    if not properties:
        return "\n".join(header + [f"data class {name}"])

    fields: List[str] = []
    for field in sorted(properties):
        prop = properties[field] or {}
        kotlin_type_name = kotlin_type(prop, root)
        if field in required:
            declaration = f"val {name_of(field)}: {kotlin_type_name},"
        else:
            # An optional field that is not given a default may be absent from the
            # response, so its type has to admit null. Without this the emitted
            # data class does not compile.
            declaration = (
                f"val {name_of(field)}: {kotlin_type_name}? = null,"
            )
        fields.append(f'    @Json(name = "{field}") {declaration}')

    # Trailing commas are valid in Kotlin and keep diffs small when fields are added.
    return "\n".join(header + [f"data class {name}("] + fields + [")"])


def emit_api(root: Dict[str, Any]) -> str:
    paths = root.get("paths", {})
    methods: List[str] = [
        "package " + PACKAGE,
        "",
        "import retrofit2.Response",
        "import retrofit2.http.Body",
        "import retrofit2.http.GET",
        "import retrofit2.http.POST",
        "import retrofit2.http.Query",
        "import retrofit2.http.Multipart",
        "import retrofit2.http.Part",
        "import okhttp3.RequestBody",
        "",
        "/**",
        " * Generated from the API contract. Do not edit by hand.",
        " *",
        " * A renamed or removed server field is caught by verifyApiContract rather",
        " * than at runtime, which is the point: the failure that matters here is",
        " * a client that compiles against a schema nobody serves, returning empty",
        " * answers with no visible error.",
        " */",
        "interface PalliSahayakApi {",
    ]

    components = (root.get("components") or {}).get("schemas") or {}
    needed: List[str] = []

    for verb, path, fn_name in CLIENT_OPERATIONS:
        operation = (paths.get(path) or {}).get(verb.lower())
        if not operation:
            continue

        params: List[str] = []
        query_args: List[str] = []
        methods_marks: List[str] = []
        for parameter in operation.get("parameters") or []:
            pname = parameter.get("name")
            ptype = kotlin_type(parameter.get("schema") or {}, root)
            pin = parameter.get("in")
            annotation = f'@Query("{pname}")' if pin == "query" else f"@Path"
            params.append(f"{annotation} {name_of(pname)}: {ptype}")
            query_args.append(name_of(pname))

        request_body = operation.get("requestBody") or {}
        content = request_body.get("content") or {}

        if "multipart/form-data" in content:
            # The voice route uploads audio alongside query parameters. Emitting it
            # as a plain POST produced a client that could not send audio at all.
            # The body is a $ref to a named component, so it must be resolved
            # before its properties can be read.
            body = resolve(content["multipart/form-data"].get("schema") or {}, root)
            methods_marks.append("    @Multipart")
            for part in sorted(body.get("properties") or {}):
                prop = body["properties"][part] or {}
                kind = kotlin_type(prop, root)
                if kind == "String":
                    # A file part arrives as binary, not as a JSON string.
                    kind = "RequestBody"
                params.append(f'@Part("{part}") {name_of(part)}: {kind}')
        elif "application/json" in content:
            body_schema = content["application/json"].get("schema") or {}
            ref = body_schema.get("$ref", "")
            body_name = ref.rsplit("/", 1)[-1] if ref else ""
            if body_name and body_name in components:
                needed.append(body_name)
                params.append(f"@Body body: {body_name}")

        response_schema = ((operation.get("responses") or {})
                           .get("200", {})
                           .get("content", {})
                           .get("application/json", {})
                           .get("schema"))
        return_type = "Response<Unit>"
        if response_schema:
            ref = response_schema.get("$ref", "")
            name = ref.rsplit("/", 1)[-1] if ref else ""
            if name and name in components:
                needed.append(name)
                return_type = f"Response<{name}>"

        if query_args:
            query_args.append("language: String")

        args = ", ".join(params)
        route = path.replace("{", "").replace("}", "")
        methods.append("")
        methods.extend(methods_marks)
        methods.append(f"    @{verb.upper()}(")
        methods.append(f'        "{route}"')
        methods.append("    )")
        if args:
            methods.append(f"    suspend fun {fn_name}({args}): {return_type}")
        else:
            methods.append(
                f"    suspend fun {fn_name}(language: String): {return_type}"
            )

    methods.append("}")
    return "\n".join(methods)


def generate(root: Dict[str, Any]) -> Dict[str, Path]:
    components = (root.get("components") or {}).get("schemas") or {}
    paths = root.get("paths", {})

    referenced: set = set()
    for verb, path, _ in CLIENT_OPERATIONS:
        operation = (paths.get(path) or {}).get(verb.lower()) or {}
        for candidate in (
            ((operation.get("requestBody") or {}).get("content", {})
             .get("application/json", {}).get("schema")),
            ((operation.get("responses") or {}).get("200", {})
             .get("content", {}).get("application/json", {}).get("schema")),
        ):
            ref = (candidate or {}).get("$ref", "")
            if ref:
                referenced.add(ref.rsplit("/", 1)[-1])

    # Follow nested component references so a DTO is never missing a field type.
    queue = list(referenced)
    while queue:
        name = queue.pop()
        schema = components.get(name)
        if not schema:
            continue
        for prop in (schema.get("properties") or {}).values():
            ref = (prop or {}).get("$ref", "")
            if ref:
                child = ref.rsplit("/", 1)[-1]
                if child not in referenced:
                    referenced.add(child)
                    queue.append(child)

    written: Dict[str, Path] = {}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = ["package " + PACKAGE, "", "import com.squareup.moshi.Json", ""]
    for name in sorted(referenced):
        models.append(emit_data_class(name, {"$ref": f"#/components/schemas/{name}"}, root))
        models.append("")
    models_path = OUT_DIR / "Models.kt"
    models_path.write_text("\n".join(models), encoding="utf-8")
    written["Models.kt"] = models_path

    api_path = OUT_DIR / "PalliSahayakApi.kt"
    api_path.write_text(emit_api(root) + "\n", encoding="utf-8")
    written["PalliSahayakApi.kt"] = api_path

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="Exit 1 if the generated sources differ from what is committed.",
    )
    args = parser.parse_args()

    try:
        root = json.loads(CONTRACT.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(
            f"Contract not found: {CONTRACT}\n"
            "Create it with: python3 scripts/export_openapi_contract.py --write",
            file=sys.stderr,
        )
        return 1
    except json.JSONDecodeError as exc:
        # A malformed contract is a different failure from a missing one, and the
        # message should say which, or the next person debugs the wrong thing.
        print(f"Contract is not valid JSON: {CONTRACT}: {exc}", file=sys.stderr)
        return 1
    before = {name: path.read_text(encoding="utf-8") for name, path in
              ({"Models.kt": OUT_DIR / "Models.kt",
                "PalliSahayakApi.kt": OUT_DIR / "PalliSahayakApi.kt"}
               .items()) if path.exists()}

    written = generate(root)

    if args.check:
        stale = [
            name for name, path in written.items()
            if before.get(name) != path.read_text(encoding="utf-8")
        ]
        if stale:
            print(
                "GENERATED CLIENT IS STALE: "
                + ", ".join(stale)
                + "\nRegenerate with: python3 scripts/generate_kotlin_client.py",
                file=sys.stderr,
            )
            return 1
        print("Generated client is up to date.")
        return 0

    for name, path in sorted(written.items()):
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
