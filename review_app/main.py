"""FastAPI entrypoint for the 6-doctor RAG review app.

Run with:

    cd review_app
    uvicorn main:app --host 0.0.0.0 --port 8005
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import auth
import data

APP_DIR = Path(__file__).parent

SEED_USERS = [
    {"user_id": "doc1", "name": "doc1", "site": "", "pin": "0175"},
    {"user_id": "doc2", "name": "doc2", "site": "", "pin": "4152"},
    {"user_id": "doc3", "name": "doc3", "site": "", "pin": "6706"},
    {"user_id": "doc4", "name": "doc4", "site": "", "pin": "6912"},
    {"user_id": "doc5", "name": "doc5", "site": "", "pin": "7682"},
    {"user_id": "doc6", "name": "doc6", "site": "", "pin": "8860"},
]

auth.ensure_users_seeded(SEED_USERS)
data.ensure_dirs()

app = FastAPI(title="Palli Sahayak — RAG Review Console", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


def current_user(request: Request) -> Optional[dict]:
    session = auth.read_session_cookie(request.cookies.get(auth.SESSION_COOKIE))
    if not session:
        return None
    return auth.get_user(session["user_id"])


def require_user(request: Request) -> dict:
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    return user


def active_version(request: Request) -> str:
    return data.normalize_version(request.cookies.get(auth.VERSION_COOKIE))


def base_context(request: Request, user: Optional[dict]) -> dict:
    return {
        "request": request,
        "user": user,
        "active_version": active_version(request),
        "available_versions": data.KNOWN_VERSIONS,
        "admin_only_versions": list(data.ADMIN_ONLY_VERSIONS),
    }


@app.get("/", include_in_schema=False)
def root(request: Request):
    if current_user(request):
        return RedirectResponse("/dashboard", status_code=303)
    return RedirectResponse("/login", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, error: Optional[str] = None):
    ctx = base_context(request, None)
    ctx.update({"users": auth.list_users(), "error": error})
    return templates.TemplateResponse("login.html", ctx)


@app.post("/admin/set-version")
def set_version(
    request: Request,
    version: str = Form(...),
    password: Optional[str] = Form(None),
    next: Optional[str] = Form(None),
):
    """Switch the active vignette set. Admin-only versions require password."""
    if version not in data.KNOWN_VERSIONS:
        return RedirectResponse(next or "/dashboard", status_code=303)
    if version in data.ADMIN_ONLY_VERSIONS and not auth.verify_admin_password(password):
        target = next or "/dashboard"
        sep = "&" if "?" in target else "?"
        return RedirectResponse(f"{target}{sep}version_error=Invalid+admin+password", status_code=303)
    resp = RedirectResponse(next or "/dashboard", status_code=303)
    resp.set_cookie(
        auth.VERSION_COOKIE,
        version,
        httponly=False,
        samesite="lax",
        max_age=60 * 60 * 24 * 30,
        path="/",
    )
    return resp


@app.post("/login")
def do_login(
    request: Request,
    user_id: str = Form(...),
    pin: str = Form(...),
):
    if not auth.verify_pin(user_id, pin):
        return RedirectResponse("/login?error=Invalid+user+or+PIN", status_code=303)
    cookie = auth.make_session_cookie(user_id)
    resp = RedirectResponse("/dashboard", status_code=303)
    resp.set_cookie(
        auth.SESSION_COOKIE,
        cookie,
        httponly=True,
        samesite="lax",
        max_age=auth.SESSION_TTL_SECONDS,
        path="/",
    )
    return resp


@app.get("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(auth.SESSION_COOKIE, path="/")
    return resp


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, filter: str = "all", version_error: Optional[str] = None):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    version = active_version(request)
    vignettes = data.list_vignettes(version)
    all_reviews = data.list_reviews()
    by_v: dict[str, list[dict]] = {}
    for r in all_reviews:
        by_v.setdefault(r["vignette_id"], []).append(r)

    rows = []
    for v in vignettes:
        revs = by_v.get(v["vignette_id"], [])
        mine = next((r for r in revs if r["reviewer_id"] == user["user_id"]), None)
        others = [r for r in revs if r["reviewer_id"] != user["user_id"]]
        if mine:
            status = "reviewed_by_you"
            status_label = f"Reviewed by you (score {mine.get('overall_score', '?')}/10)"
        elif others:
            status = "reviewed_by_others"
            status_label = f"Reviewed by {len(others)} other(s)"
        else:
            status = "pending"
            status_label = "Pending"
        rows.append(
            {
                **v,
                "status": status,
                "status_label": status_label,
                "n_others": len(others),
                "your_score": mine.get("overall_score") if mine else None,
            }
        )

    if filter == "pending":
        rows = [r for r in rows if r["status"] == "pending"]
    elif filter == "mine":
        rows = [r for r in rows if r["status"] == "reviewed_by_you"]

    ctx = base_context(request, user)
    ctx.update(
        {
            "vignettes": rows,
            "filter": filter,
            "total": len(vignettes),
            "version_error": version_error,
            "n_mine": sum(1 for r in rows if r["status"] == "reviewed_by_you")
            if filter != "all"
            else sum(1 for v in vignettes if any(
                rv["reviewer_id"] == user["user_id"] and rv["vignette_id"] == v["vignette_id"]
                for rv in all_reviews
            )),
        }
    )
    return templates.TemplateResponse("dashboard.html", ctx)


@app.get("/review/{vignette_id}", response_class=HTMLResponse)
def review_page(request: Request, vignette_id: str):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    version = active_version(request)
    vignette = data.load_vignette(vignette_id, version)
    if not vignette:
        raise HTTPException(404, f"Vignette {vignette_id} not found")
    rag_outputs = data.load_rag_outputs(vignette_id, version)
    if not rag_outputs:
        rag_outputs = [{
            "provider": "unknown",
            "label": "(no RAG output found)",
            "output": {
                "answer": "(No RAG output found yet for this vignette.)",
                "sources": [], "evidence_level": "unknown",
                "validation_result": {},
                "language": vignette.get("language", ""),
            },
        }]
    # Back-compat: keep `rag_output` (first one) for templates that still use it.
    rag_output = rag_outputs[0]["output"]
    rubric = data.load_rubric()
    existing = data.review_by(user["user_id"], vignette_id)

    all_v = data.list_vignettes(version)
    idx = next((i for i, v in enumerate(all_v) if v["vignette_id"] == vignette_id), -1)
    position = f"{idx + 1} of {len(all_v)}" if idx >= 0 else f"? of {len(all_v)}"

    ctx = base_context(request, user)
    ctx.update(
        {
            "vignette": vignette,
            "rag_output": rag_output,
            "rag_outputs": rag_outputs,
            "rubric": rubric,
            "existing": existing,
            "position": position,
        }
    )
    return templates.TemplateResponse("review.html", ctx)


@app.post("/review/{vignette_id}")
async def submit_review(request: Request, vignette_id: str):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    version = active_version(request)
    vignette = data.load_vignette(vignette_id, version)
    if not vignette:
        raise HTTPException(404, f"Vignette {vignette_id} not found")

    form = await request.form()
    rubric = data.load_rubric()

    dimension_scores = {}
    for dim in rubric["dimensions"]:
        raw = form.get(f"dim_{dim['key']}")
        if raw is None or raw == "":
            raise HTTPException(400, f"Missing score for {dim['label']}")
        try:
            dimension_scores[dim["key"]] = int(raw)
        except ValueError as e:
            raise HTTPException(400, f"Invalid score for {dim['label']}") from e

    sub_item_scores = {}
    for dim in rubric["dimensions"]:
        for item in dim["sub_items"]:
            val = form.get(f"sub_{item['key']}")
            if val == "yes":
                sub_item_scores[item["key"]] = True
            elif val == "no":
                sub_item_scores[item["key"]] = False
            else:
                sub_item_scores[item["key"]] = None

    try:
        overall_score = int(form.get("overall_score", "0"))
    except ValueError:
        overall_score = 0
    overall_score = max(0, min(10, overall_score))

    reviewer_real_name = (form.get("reviewer_real_name") or "").strip()
    reviewer_institution = (form.get("reviewer_institution") or "").strip()
    if not reviewer_real_name:
        raise HTTPException(400, "Your name is required.")
    if not reviewer_institution:
        raise HTTPException(400, "Your institution is required.")

    comments = (form.get("comments") or "").strip()
    if len(comments) < 50:
        raise HTTPException(400, "Comments must be at least 50 characters.")
    corrections = (form.get("corrections") or "").strip()
    tags = form.getlist("tags") if hasattr(form, "getlist") else form.get("tags", [])
    if isinstance(tags, str):
        tags = [tags] if tags else []
    recommend = form.get("would_recommend_for_clinical_use", "with_revisions")
    flag = form.get("flag_for_committee_review") in ("on", "true", "1", "yes")
    try:
        time_seconds = int(form.get("time_to_review_seconds", "0"))
    except ValueError:
        time_seconds = 0

    review = {
        "reviewer_id": user["user_id"],
        "reviewer_name": user["name"],
        "reviewer_real_name": reviewer_real_name,
        "reviewer_institution": reviewer_institution,
        "vignette_id": vignette_id,
        "vignette_version": version,
        "dimension_scores": dimension_scores,
        "sub_item_scores": sub_item_scores,
        "overall_score": overall_score,
        "comments": comments,
        "corrections": corrections,
        "tags": list(tags),
        "would_recommend_for_clinical_use": recommend,
        "flag_for_committee_review": flag,
        "time_to_review_seconds": time_seconds,
    }
    data.save_review(review)
    return RedirectResponse("/dashboard", status_code=303)


@app.get("/progress", response_class=HTMLResponse)
def progress_view(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    version = active_version(request)
    reviewers = auth.list_users()
    vignettes = data.list_vignettes(version)
    matrix = data.progress_matrix(reviewers, vignettes)

    per_reviewer_totals = {
        r["user_id"]: sum(1 for v in vignettes if matrix[r["user_id"]][v["vignette_id"]])
        for r in reviewers
    }
    per_vignette_totals = {
        v["vignette_id"]: sum(1 for r in reviewers if matrix[r["user_id"]][v["vignette_id"]])
        for v in vignettes
    }
    ctx = base_context(request, user)
    ctx.update(
        {
            "reviewers": reviewers,
            "vignettes": vignettes,
            "matrix": matrix,
            "per_reviewer_totals": per_reviewer_totals,
            "per_vignette_totals": per_vignette_totals,
        }
    )
    return templates.TemplateResponse("progress.html", ctx)


@app.get("/export.csv")
def export_csv(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    csv_text = data.export_reviews_csv()
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="expert_reviews.csv"'},
    )


# ---------- Admin ----------


def is_admin(request: Request) -> bool:
    return auth.read_admin_cookie(request.cookies.get(auth.ADMIN_SESSION_COOKIE))


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request, error: Optional[str] = None):
    user = current_user(request)
    if not is_admin(request):
        ctx = base_context(request, user)
        ctx["error"] = error
        return templates.TemplateResponse("admin_login.html", ctx)

    reviewers = auth.list_users()
    summary_map = data.per_reviewer_summary()
    per_version_total = {v: len(data.list_vignettes(v)) for v in data.KNOWN_VERSIONS}

    rows = []
    for u in reviewers:
        s = summary_map.get(
            u["user_id"],
            {
                "reviewer_id": u["user_id"],
                "reviewer_name": u["name"],
                "real_name": "",
                "institution": "",
                "by_version": {v: 0 for v in data.KNOWN_VERSIONS},
                "total": 0,
                "avg_overall": None,
                "last_submitted_at": "",
            },
        )
        s["reviewer_name"] = u["name"]
        rows.append(s)

    grand_total = sum(r["total"] for r in rows)
    grand_target = sum(per_version_total.values()) * len(reviewers)

    ctx = base_context(request, user)
    ctx.update(
        {
            "rows": rows,
            "versions": data.KNOWN_VERSIONS,
            "per_version_total": per_version_total,
            "grand_total": grand_total,
            "grand_target": grand_target,
        }
    )
    return templates.TemplateResponse("admin.html", ctx)


@app.post("/admin/login")
def admin_login(password: str = Form(...)):
    if not auth.verify_admin_password(password):
        return RedirectResponse("/admin?error=Invalid+admin+password", status_code=303)
    resp = RedirectResponse("/admin", status_code=303)
    resp.set_cookie(
        auth.ADMIN_SESSION_COOKIE,
        auth.make_admin_cookie(),
        httponly=True,
        samesite="lax",
        max_age=auth.SESSION_TTL_SECONDS,
        path="/",
    )
    return resp


@app.get("/admin/logout")
def admin_logout():
    resp = RedirectResponse("/admin", status_code=303)
    resp.delete_cookie(auth.ADMIN_SESSION_COOKIE, path="/")
    return resp


@app.get("/admin/export-all.csv")
def admin_export_all(request: Request):
    if not is_admin(request):
        return RedirectResponse("/admin", status_code=303)
    return Response(
        content=data.export_reviews_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="all_reviews_all_doctors.csv"'},
    )


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {
        "status": "ok",
        "reviewers": len(auth.list_users()),
        "vignettes": {v: len(data.list_vignettes(v)) for v in data.KNOWN_VERSIONS},
        "default_version": data.DEFAULT_VERSION,
    }
