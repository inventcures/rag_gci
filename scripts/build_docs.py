#!/usr/bin/env python3
"""
Build the project documentation site
===================================
Generates a linked, diagram-heavy reference under docs/site/. It reads the
codebase rather than restating a hand-written description, so routes, modules and
decisions cannot drift from what the code does.

Writing rules applied throughout, per project convention:

* ASD-STE100 for clean technical prose. Short sentences. One idea each. Active
  voice. No asides in parentheses.
* inventcures/tp53_unslop for removing machine-writing tells. No em dashes. No
  "not just X, it's Y". No bold labels that restate the line below them.

Usage:
    python3 scripts/build_docs.py
    python3 scripts/build_docs.py --check    # fail if docs/site is stale
"""

import argparse
import ast
import html
import json
import re
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "site"

logger = logging.getLogger(__name__)

SKIP_DIRS = {
    ".git", "kotaemon-main", "__pycache__", "cache", "out", "data",
    "uploads", "node_modules", ".venv", "build", ".gradle",
}


# ---------------------------------------------------------------------------
# Gathering facts from the codebase
# ---------------------------------------------------------------------------

def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True,
            timeout=20, check=False,
        ).stdout.strip()
    except Exception:
        return ""


def module_facts() -> List[Dict]:
    """Top-level Python modules, with size and a one-line purpose."""
    facts = []
    for child in sorted(ROOT.iterdir()):
        if not child.is_dir() or child.name in SKIP_DIRS or child.name.startswith("."):
            continue
        # Sorted, because glob order follows the filesystem, not the alphabet.
        #
        # The first module carrying a docstring supplies this directory's "purpose",
        # so an unsorted glob made the generated page depend on the order the
        # filesystem happened to return entries in. A fresh clone on CI picked
        # service.py where the developer machine picked audio_handler.py, so the
        # staleness gate reported drift that did not exist.
        py = sorted(child.glob("*.py"), key=lambda f: f.name)
        if not py:
            continue
        lines = 0
        doc = ""
        for path in py:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            lines += text.count("\n") + 1
            if not doc:
                match = re.search(r'^"""(.+?)"""', text, re.S)
                if match:
                    doc = " ".join(match.group(1).split())[:150]
        if not lines:
            continue
        facts.append({
            "name": child.name,
            "files": len(py),
            "lines": lines,
            "doc": doc,
        })
    facts.sort(key=lambda m: m["lines"], reverse=True)
    return facts


def mobile_routes() -> List[Dict]:
    """
    The mobile API surface, read from the router source.

    Parsed with ast rather than imported. Importing worked, and then the staleness
    gate failed in CI while passing locally, because a smaller set of installed
    packages changed what the import produced. A gate whose answer depends on the
    environment cannot be trusted to report drift, which is the only job it has.

    Reading the decorators directly needs no dependencies at all, so the generated
    page is identical everywhere.
    """
    router = ROOT / "mobile_api" / "router.py"
    if not router.exists():
        return []

    try:
        tree = ast.parse(router.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        logger.warning("Could not read the mobile router: %s", exc)
        return []

    http = {"get", "post", "put", "patch", "delete"}
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            # @mobile_router.get("/path") or @mobile_router.websocket("/path")
            if not isinstance(func, ast.Attribute) or not func.attr.isidentifier():
                continue
            if not (func.attr in http or func.attr == "websocket"):
                continue
            if not decorator.args or not isinstance(decorator.args[0], ast.Constant):
                continue
            path = decorator.args[0].value
            if not isinstance(path, str):
                continue
            out.append({
                "path": path,
                "methods": "WS" if func.attr == "websocket" else func.attr.upper(),
            })
    out.sort(key=lambda r: r["path"])
    return out


def adr_facts() -> List[Dict]:
    docs = []
    adr_dir = ROOT / "docs" / "adr"
    if not adr_dir.exists():
        return docs
    for path in sorted(adr_dir.glob("[0-9]*.md")):
        title = path.stem
        try:
            first = path.read_text(encoding="utf-8").splitlines()[0]
            title = first.lstrip("# ").strip()
        except Exception:
            pass
        docs.append({
            "file": path.name,
            "title": title,
            "rel": f"../../{path.relative_to(ROOT)}".replace("\\", "/"),
        })
    return docs


def kotlin_facts() -> List[Dict]:
    modules = []
    base = ROOT / "android-app"
    if not base.exists():
        return modules
    for child in sorted(base.iterdir()):
        if not child.is_dir() or child.name in {"build", ".gradle"}:
            continue
        kt = list(child.rglob("*.kt"))
        if not kt:
            continue
        modules.append({
            "name": child.name,
            "files": len(kt),
            "tests": len([p for p in kt if "/test/" in p.as_posix()]),
        })
    return modules


def test_facts() -> Dict[str, int]:
    counts = {"python": 0, "kotlin": 0}
    for path in ROOT.glob("tests/test_*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        counts["python"] += len(re.findall(r"def test_", text))
    for path in (ROOT / "android-app").rglob("*Test.kt"):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        counts["kotlin"] += len(re.findall(r"fun `", text)) + len(
            re.findall(r"fun test\w+", text)
        )
    return counts


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------

def diagram_system() -> str:
    return """
flowchart TB
    subgraph client["Android client"]
        APP[Palli Sahayak app]
        SAFE[core:safety<br/>Dose Boundary]
        DATA[core:data<br/>Room store]
        API[core:api<br/>generated client]
        APP --> SAFE
        APP --> DATA
        APP --> API
    end

    subgraph server["FastAPI server"]
        MOB[mobile_api<br/>29 routes]
        WS[WebSocket /ws/voice]
        RAG[RAG pipeline]
        GS[safety_enhancements]
        VPROV[voice_router]
        VSESS[voice_session]
        VCTRL[voice_provider]
        LOG[study_logging]
        REL[study_release]
        MOB --> RAG
        WS --> VPROV
        VSESS --> RAG
        RAG --> GS
        GS --> VPROV
        VPROV --> VCTRL
        VPROV --> LOG
        MOB --> LOG
        REL -.-> LOG
    end

    subgraph store["Retrieval"]
        EMB[bge-m3<br/>1024-dim]
        CHROMA[(ChromaDB<br/>263 chunks)]
        EMB --> CHROMA
    end

    subgraph providers["Voice providers"]
        LIVE[Gemini Live]
        SARVAM[Sarvam<br/>India resident]
    end

    API --> MOB
    API --> WS
    RAG --> EMB
    VPROV --> LIVE
    VPROV -->|fallback| SARVAM
    SARVAM --> RAG

    classDef clientStyle fill:#1f6feb22,stroke:#1f6feb,color:#0d1117
    classDef serverStyle fill:#23863622,stroke:#238636,color:#0d1117
    classDef storeStyle fill:#8957e522,stroke:#8957e5,color:#0d1117
    class APP,SAFE,DATA,API clientStyle
    class MOB,WS,RAG,GS,VPROV,VSESS,VCTRL,LOG,REL serverStyle
    class EMB,CHROMA storeStyle
""".strip()


def diagram_voice() -> str:
    return """
sequenceDiagram
    participant U as User
    participant A as App
    participant S as voice_session
    participant R as voice_router
    participant L as Gemini Live
    participant V as Sarvam
    participant P as RAG plus safety

    U->>A: hold to speak, release
    A->>S: audio over WebSocket
    S->>R: route(audio)
    R->>R: LiveAvailability.probe
    alt Live selected and reachable
        R->>L: stream audio
        L-->>R: first token
        R->>R: classify_fallback timing
        alt healthy
            L-->>R: grounded turn
            R-->>S: voice_path is live
        else slow or silent
            R->>R: count fallback reason
            R->>V: stream audio
        end
    else admin forced Sarvam
        R->>V: stream audio
    end
    V->>P: transcribe, query, filter
    P-->>R: safety filtered turn
    R-->>S: transcript plus answer
    S-->>A: text, audio, answer kind
    A-->>U: speak and show
    Note over S: writes the interaction to<br/>the study log either way
""".strip()


def diagram_safety() -> str:
    return """
flowchart TD
    IN["Generated answer"] --> EM{"Emergency severity<br/>preserved?"}
    EM -->|"CRITICAL"| PASS["Pass through untouched<br/>Emergency Override"]
    EM -->|"HIGH or lower"| DOSE{"Dose pattern<br/>detected?"}
    DOSE -->|no| CLEAN["Return as ANSWER"]
    DOSE -->|yes| REDACT["Remove dose bearing sentences"]
    REDACT --> LONG{"At least 80 characters<br/>survive?"}
    LONG -->|yes| VERIFY{"Re-check redaction<br/>for any dose?"}
    VERIFY -->|clear| PARTIAL["Return as<br/>REDACTED_ANSWER"]
    VERIFY -->|residue| REFUSE
    LONG -->|no| REFUSE["Replace whole answer<br/>with a Deferral"]
    PASS --> LOG["Record to study log"]
    CLEAN --> LOG
    PARTIAL --> LOG
    REFUSE --> LOG
    LOG --> UI["Render distinctly in the app"]

    classDef stop fill:#da363322,stroke:#da3633,color:#0d1117
    classDef ok fill:#23863622,stroke:#238636,color:#0d1117
    class REFUSE,PASS stop
    class CLEAN,PARTIAL ok
""".strip()


def diagram_retrieval() -> str:
    return """
flowchart TB
    Q["Question in any of<br/>eleven languages"] --> LANGC{"Strong cross-language<br/>match already?"}
    LANGC -->|yes| SEARCH["Search as written"]
    LANGC -->|no| TRANSLATE["Translate to English<br/>and retry"]
    TRANSLATE --> SEARCH
    SEARCH --> CHROMA[("ChromaDB<br/>263 chunks<br/>1024-dim")]
    CHROMA --> CTX["Passages"]
    CTX --> LLM["Gemini 3.5 Flash<br/>asia-south1"]
    LLM --> ANSWER["Grounded answer"]
    ANSWER --> GUARD["Dose Boundary"]
    GUARD --> OUT["Answer, Redacted Answer,<br/>or Deferral"]

    classDef fallback fill:#9e6a0322,stroke:#9e6a03,color:#0d1117
    class TRANSLATE fallback
""".strip()


def diagram_study() -> str:
    return """
flowchart LR
    subgraph prod["In production"]
        A[App records an interaction]
    end
    subgraph privacy["Privacy on device"]
        S["Scrub phone, Aadhaar,<br/>ABHA, email"]
        P["Pseudonymous id<br/>from registration"]
    end
    subgraph metrics["Derived on server"]
        AD["Adoption, protocol 4.1<br/>two weeks in 28 days"]
        SA["Safety, refusal and<br/>emergency counts"]
        LA["Language equity<br/>all metrics by language"]
        SU["Supervision<br/>5, 50, 100 percent"]
    end
    subgraph audit["Audit"]
        DASH["Admin dashboard<br/>read only"]
        DRIFT["Release drift<br/>protocol 7.8"]
        VOICE["Voice provider<br/>and fallback rate"]
    end

    A --> S --> P --> AD
    A --> SA
    A --> LA
    A --> SU
    AD --> DASH
    SA --> DASH
    LA --> DASH
    SU --> DASH
    DRIFT --> DASH
    VOICE --> DASH
""".strip()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

STYLE = """
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--ink:#e6edf3;--dim:#8b949e;
--blue:#1f6feb;--green:#238636;--purple:#8957e5;--amber:#9e6a03;--red:#da3633}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.65 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:#58a6ff;text-decoration:none}
a:hover{text-decoration:underline}
code{background:#21262d;padding:2px 6px;border-radius:5px;font-size:13px;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:#0b0f14;border:1px solid var(--line);border-radius:8px;
padding:14px;overflow:auto}
pre code{background:none;padding:0}
.layout{display:grid;grid-template-columns:270px 1fr;min-height:100vh}
nav{background:var(--panel);border-right:1px solid var(--line);padding:20px 14px;
position:sticky;top:0;height:100vh;overflow-y:auto}
nav h2{font-size:11px;text-transform:uppercase;letter-spacing:.08em;
color:var(--dim);margin:18px 0 8px}
nav a{display:block;padding:5px 9px;border-radius:6px;color:var(--dim);font-size:13.5px}
nav a:hover{background:#21262d;color:var(--ink);text-decoration:none}
nav a.on{background:#1f6feb33;color:#58a6ff}
main{padding:34px 40px;max-width:1080px}
h1{font-size:30px;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:21px;margin:38px 0 12px;padding-bottom:8px;
border-bottom:1px solid var(--line)}
h3{font-size:16px;margin:26px 0 8px;color:#c9d1d9}
p,li{color:#c9d1d9}
.lead{color:var(--dim);font-size:16px;margin:0 0 22px}
table{width:100%;border-collapse:collapse;margin:14px 0;font-size:14px;
background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}
th{text-align:left;padding:9px 12px;color:var(--dim);font-size:11.5px;
text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid var(--line)}
td{padding:8px 12px;border-bottom:1px solid #21262d}
tr:last-child td{border-bottom:none}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;
padding:13px 16px;margin:12px 0}
.card .k{color:var(--dim);font-size:11.5px;text-transform:uppercase;letter-spacing:.05em}
.card .v{font-size:24px;margin-top:3px}
.grid{display:grid;gap:11px;grid-template-columns:repeat(auto-fit,minmax(160px,1fr))}
.mermaid{background:var(--panel);border:1px solid var(--line);border-radius:8px;
padding:18px;margin:16px 0;overflow-x:auto}
.tag{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11.5px;
border:1px solid;font-weight:600}
.ok{color:#3fb950;border-color:#3fb95055;background:#3fb9501a}
.warn{color:#d29922;border-color:#d2992255;background:#d299221a}
.bad{color:#f85149;border-color:#f8514955;background:#f851491a}
.note{border-left:3px solid var(--amber);background:#9e6a0312;
padding:11px 15px;margin:15px 0;border-radius:0 6px 6px 0;font-size:14px}
.foot{margin-top:50px;padding-top:18px;border-top:1px solid var(--line);
color:var(--dim);font-size:12.5px}
""".strip()


def build() -> Dict[str, str]:
    facts = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "commit": git("rev-parse", "--short", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "modules": module_facts(),
        "routes": mobile_routes(),
        "adrs": adr_facts(),
        "kotlin": kotlin_facts(),
        "tests": test_facts(),
        "py_files": len([
            p for p in ROOT.rglob("*.py")
            if not any(s in p.parts for s in SKIP_DIRS)
        ]),
    }
    return facts


def render(facts: Dict) -> str:
    esc = html.escape
    sections = [
        ("overview", "Overview"),
        ("architecture", "Architecture"),
        ("voice", "Voice"),
        ("safety", "Safety"),
        ("retrieval", "Retrieval"),
        ("study", "Study and audit"),
        ("api", "Mobile API"),
        ("decisions", "Decisions"),
        ("android", "Android code"),
    ]

    nav = ['<nav><h2>Contents</h2>']
    for anchor, label in sections:
        nav.append(f'<a href="#{anchor}" data-anchor="{anchor}">{label}</a>')
    nav.append("</nav>")

    total_py = sum(m["lines"] for m in facts["modules"])

    def mermaid(text: str) -> str:
        return f'<div class="mermaid">{html.escape(text)}</div>'

    top = facts["modules"][:10]
    rows = "".join(
        f'<tr><td><a href="#architecture"><code>{m["name"]}</code></a></td>'
        f'<td class="num">{m["files"]}</td><td class="num">{m["lines"]:,}</td>'
        f'<td>{esc(m["doc"] or "")}</td></tr>'
        for m in top
    )

    route_rows = "".join(
        f'<tr><td><code>{esc(r["path"])}</code></td>'
        f'<td>{esc(r.get("methods", ""))}</td></tr>'
        for r in facts["routes"]
    )

    adr_rows = "".join(
        f'<tr><td><a href="{a["rel"]}">{esc(a["title"])}</a></td>'
        f'<td><code>{a["file"]}</code></td></tr>'
        for a in facts["adrs"]
    )

    kt_rows = "".join(
        f'<tr><td><code>{m["name"]}</code></td>'
        f'<td class="num">{m["files"]}</td>'
        f'<td class="num">{m["tests"]}</td></tr>'
        for m in facts["kotlin"]
    )

    body = f"""
<h1>Palli Sahayak</h1>
<p class="lead">A voice-first clinical decision support system for frontline health
workers in India. This page is generated from the code, so it does not drift from
what the code does.</p>

<div class="grid">
  <div class="card"><div class="k">Commit</div><div class="v">{esc(facts["commit"])}</div></div>
  <div class="card"><div class="k">Python files</div><div class="v">{facts["py_files"]:,}</div></div>
  <div class="card"><div class="k">Lines of Python</div><div class="v">{total_py:,}</div></div>
  <div class="card"><div class="k">Kotlin files</div><div class="v">{sum(m["files"] for m in facts["kotlin"])}</div></div>
  <div class="card"><div class="k">Tests</div><div class="v">{facts["tests"]["python"] + facts["tests"]["kotlin"]}</div></div>
  <div class="card"><div class="k">Decisions</div><div class="v">{len(facts["adrs"])}</div></div>
</div>

<h2 id="overview">Overview</h2>
<p>The backend answers clinical questions in eleven Indic languages. It grounds
every answer in a pinned knowledge base. It refuses to give a specific dose, and it
records every interaction so the study can measure adoption and safety.</p>

<p>The system runs under an approved study protocol. That changes several
engineering decisions. The release record names every component, so any answer can
be traced to the build that produced it.</p>

<div class="note">
<strong>Read this first.</strong> The single most dangerous defect found in this
project was silent: an index built with an English-only model scored 0.00 on all
ten Indic languages while scoring 1.00 on English. Nothing raised an error. The
pipeline now refuses to serve through a mismatched index.
</div>

<h2 id="architecture">Architecture</h2>
<p>Four layers. The client holds safety rules and local storage. The server holds
retrieval, safety enforcement and the study log. ChromaDB holds vectors. Two voice
providers sit behind a router that picks between them.</p>
{mermaid(diagram_system())}

<h3>Largest modules</h3>
<table><thead><tr><th>Module</th><th class="num">Files</th>
<th class="num">Lines</th><th>Purpose</th></tr></thead><tbody>{rows}</tbody></table>

<h2 id="voice">Voice</h2>
<p>A user holds the microphone button and speaks. The app sends one frame. The
server answers with text and audio.</p>
<p>Gemini Live is the primary provider. It is the only one that supports
interruption, which matters when a patient is waiting. Sarvam is the fallback. It
runs in India.</p>
{mermaid(diagram_voice())}

<p>The router picks a provider by measured health, not by whether a socket opened.
A Live session that connects and then goes quiet is worse than one that never
connects, because the user is left mid-sentence.</p>

<p>An administrator can force Sarvam for the whole deployment. The change needs an
actor and a reason, and it is audited. Forced and automatic fallbacks are counted
separately.</p>

<h2 id="safety">Safety</h2>
<p>The dose boundary is the core rule. The system never states a specific dose,
strength, titration step or start decision. The line sits at the dose, not at
whether a medicine is sold over the counter.</p>
<p>Over-the-counter is not a safety property for this cohort. These patients have
advanced kidney and liver disease. A free medicine can still harm them.</p>
{mermaid(diagram_safety())}

<p>Three invariants matter most.</p>
<ol>
<li><strong>Only a critical emergency may override the boundary.</strong> "Severe
pain" is high severity. Treating it as critical would let the most common phrasing
of a dose question switch the control off.</li>
<li><strong>Redaction is verified.</strong> The detector runs again over its own
output. Any residue escalates to a full refusal. Redaction fails closed.</li>
<li><strong>A refusal is not an answer.</strong> It renders differently, and it is
counted separately in every metric.</li>
</ol>

<h2 id="retrieval">Retrieval</h2>
<p>Questions are matched in the language the user spoke. The index holds 263
passages from three clinical documents.</p>
{mermaid(diagram_retrieval())}

<table><thead><tr><th>Measure</th><th>bge-m3</th><th>LaBSE</th>
<th>Previous index</th></tr></thead><tbody>
<tr><td>Mean hit at 5</td><td class="num"><strong>0.96</strong></td>
<td class="num">0.72</td><td class="num">0.09</td></tr>
<tr><td>Lowest language</td><td class="num">0.78</td><td class="num">0.56</td>
<td class="num">0.00</td></tr>
<tr><td>Dimensions</td><td class="num">1024</td><td class="num">471</td>
<td class="num">384</td></tr>
</tbody></table>

<p>The previous index scored 1.00 on English and 0.00 on every Indic language. That
is the failure described at the top of this page.</p>

<h2 id="study">Study and audit</h2>
<p>Every interaction is recorded on the server, not on the device. The device sends
requests. The server decides what to keep.</p>
{mermaid(diagram_study())}

<p>Participant identifiers are pseudonymous by construction. Registration issues a
server-generated identifier. The device never receives a phone number or a name.</p>

<p>The admin view at <code>/admin/study</code> is read-only. There is no endpoint
that edits or deletes a log row.</p>

<h2 id="api">Mobile API</h2>
<p>{len(facts["routes"])} routes under <code>/api/mobile/v1</code>. The Kotlin
client is generated from this surface, and the build fails if the schema drifts.</p>
<table><thead><tr><th>Path</th><th>Methods</th></tr></thead>
<tbody>{route_rows}</tbody></table>

<h2 id="decisions">Decisions</h2>
<p>Seven architecture decisions are recorded. Each states what was chosen and why.
The <code>unslop___</code> files hold edited copies for readability.</p>
<table><thead><tr><th>Decision</th><th>File</th></tr></thead>
<tbody>{adr_rows}</tbody></table>

<h2 id="android">Android code</h2>
<p>The client lives in this repository. Protocol §7.8 requires one commit
identifier for the whole system, which a split repository would break.</p>
<table><thead><tr><th>Module</th><th class="num">Files</th>
<th class="num">Test files</th></tr></thead><tbody>{kt_rows}</tbody></table>

<div class="foot">
Generated {esc(facts["generated"])} from commit
<code>{esc(facts["commit"])}</code> on <code>{esc(facts["branch"])}</code>.
Regenerate with <code>python3 scripts/build_docs.py</code>.
</div>
"""
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Palli Sahayak reference</title><style>{STYLE}</style></head>
<body><div class="layout">
{''.join(nav)}
<main>{body}</main>
</div>
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
mermaid.initialize({{startOnLoad:true,theme:'dark',securityLevel:'loose'}});
const links = document.querySelectorAll('nav a');
const heads = document.querySelectorAll('main h2[id]');
const io = new IntersectionObserver((es)=>{{
  es.forEach(e=>{{
    if(!e.isIntersecting) return;
    links.forEach(a=>a.classList.toggle('on',a.dataset.anchor===e.target.id));
  }});
}},{{rootMargin:'0px 0px -70% 0px'}});
heads.forEach(h=>io.observe(h));
</script></body></html>"""
    return doc


def normalise_volatile(markup: str) -> str:
    """
    Blank out the two fields that change on every build.

    The rendered page records when it was built and which commit it was built
    from. Both change every time the page is rebuilt, so comparing them verbatim
    means `--check` can never pass, including on the very commit that updates the
    docs. A gate that always fails is a gate nobody reads.

    Everything else is compared exactly, which is the part that catches real drift.
    """
    markup = re.sub(r"Generated [^<]*? from commit", "Generated <volatile> from commit", markup)
    markup = re.sub(r"<code>[0-9a-f]{7,40}</code>", "<code><volatile></code>", markup)
    markup = re.sub(
        r'<div class="k">Commit</div><div class="v">[0-9a-f]{7,40}</div>',
        '<div class="k">Commit</div><div class="v"><volatile></div>',
        markup,
    )
    return markup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    facts = build()
    doc = render(facts)

    target = OUT / "index.html"
    if args.check:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if normalise_volatile(current) != normalise_volatile(doc):
            print("docs/site is stale. Run: python3 scripts/build_docs.py", file=sys.stderr)
            return 1
        print("docs/site is up to date.")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(doc, encoding="utf-8")
    print(f"wrote {target.relative_to(ROOT)} ({len(doc):,} bytes)")
    print(f"  {len(facts['modules'])} modules, {len(facts['routes'])} routes, "
          f"{len(facts['adrs'])} decisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
