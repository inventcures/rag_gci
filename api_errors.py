"""
One shape for every error response: {"status": "error", "error": "<message>"}.

AGENTS.md prescribes that shape. FastAPI's defaults answer an HTTPException
with {"detail": ...}, a bad body with a list of validation errors, and an
unhandled exception with plain text; routes also used to add a "traceback"
field. `install_error_handlers` makes the first three use the one shape. An
unhandled exception is logged server-side with its traceback and answered
with a generic message, so no internal detail reaches the caller.
"""

import logging
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

INTERNAL_ERROR_MESSAGE = "Internal server error"


def error_body(message: str, **extra: Any) -> Dict[str, Any]:
    """The one error shape, with optional extra fields (never a traceback)."""
    return {"status": "error", "error": message, **extra}


async def _http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """HTTPException keeps its status and headers (WWW-Authenticate on 401)."""
    detail = exc.detail if isinstance(exc.detail, str) else jsonable_encoder(exc.detail)
    message = detail if isinstance(detail, str) else "Request failed"
    extra = {} if isinstance(detail, str) else {"details": detail}
    return JSONResponse(error_body(message, **extra), status_code=exc.status_code,
                        headers=getattr(exc, "headers", None))


async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """A body or query that fails validation; the field list says which."""
    fields = [
        {"loc": list(err.get("loc", ())), "msg": err.get("msg", "")}
        for err in exc.errors()
    ]
    return JSONResponse(error_body("Invalid request", details=fields), status_code=422)


async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Log the traceback here; send the caller nothing about it."""
    logger.error("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(error_body(INTERNAL_ERROR_MESSAGE), status_code=500)


def install_error_handlers(app: FastAPI) -> None:
    """Register the one-shape handlers on `app`."""
    app.add_exception_handler(StarletteHTTPException, _http_exception)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unhandled)
