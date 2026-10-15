"""
Webhook verification for Bolna, Retell, Sarvam and Twilio.

Every provider webhook is accepted only with a valid signature (and, for
Bolna, its Bearer key too), and every one fails closed: when its secret is not
configured the request is refused, never waved through. Secrets are read from
the environment on each request, so a deployment that rotates them needs no
restart.

Each `require_*` function is a FastAPI dependency; a route uses it as
`@app.post(path, dependencies=[Depends(require_bolna)])`.

  * Bolna: `X-Bolna-Signature` = hex HMAC-SHA256 of the body with
    BOLNA_WEBHOOK_SECRET, and `Authorization: Bearer <RAG_API_KEY>` (the key
    bolna_integration/config.py puts in the agent's custom function).
  * Retell: `X-Retell-Signature` = `v=<ms>,d=<hex HMAC-SHA256(body + ms)>`
    keyed by RETELL_API_KEY (the retell-sdk scheme), within five minutes.
  * Sarvam: `X-Sarvam-Signature` = hex HMAC-SHA256 of the body with
    SARVAM_WEBHOOK_SECRET.
  * Twilio: `X-Twilio-Signature` = base64 HMAC-SHA1 of the public URL plus the
    sorted form key+value pairs, keyed by TWILIO_AUTH_TOKEN. The URL is built
    from PUBLIC_BASE_URL when set, because behind ngrok or a proxy the URL the
    server sees is not the one Twilio signed.

Rejections: 401 for a missing or wrong signature or key, 403 when the secret
is not configured on this server.
"""

import base64
import hashlib
import hmac
import logging
import os
import re
import time
from typing import Iterable, Optional, Tuple

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

RETELL_TOLERANCE_MS = 5 * 60 * 1000
_RETELL_SIGNATURE = re.compile(r"^v=(\d+),d=([0-9a-fA-F]+)$")


def _hex_hmac_sha256(secret: str, payload: bytes) -> str:
    """Hex HMAC-SHA256 of payload keyed by secret."""
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def _equal(expected: str, given: Optional[str]) -> bool:
    """Constant-time comparison that treats a missing value as a mismatch."""
    return bool(given) and hmac.compare_digest(expected.encode(), given.strip().encode())


# ---------------------------------------------------------------------------
# Pure checks
# ---------------------------------------------------------------------------


def verify_hmac_sha256(secret: str, body: bytes, signature: Optional[str]) -> bool:
    """True when signature is the hex HMAC-SHA256 of body; False when no secret."""
    if not secret:
        return False
    return _equal(_hex_hmac_sha256(secret, body), signature)


def verify_bearer(expected_key: str, authorization: Optional[str]) -> bool:
    """True when the Authorization header is `Bearer <expected_key>`; False when no key."""
    if not expected_key or not authorization:
        return False
    scheme, _, token = authorization.partition(" ")
    return scheme.lower() == "bearer" and _equal(expected_key, token)


def verify_retell(api_key: str, body: bytes, signature: Optional[str],
                  now_ms: Optional[int] = None) -> bool:
    """Retell's scheme: `v=<ms>,d=<hex HMAC-SHA256(body + ms)>` keyed by the API key."""
    if not api_key or not signature:
        return False
    match = _RETELL_SIGNATURE.match(signature.strip())
    if not match:
        return False
    stamp, digest = match.groups()
    now = int(time.time() * 1000) if now_ms is None else now_ms
    if abs(now - int(stamp)) > RETELL_TOLERANCE_MS:
        return False
    return _equal(_hex_hmac_sha256(api_key, body + stamp.encode()), digest.lower())


def verify_twilio(auth_token: str, url: str, params: Iterable[Tuple[str, str]],
                  signature: Optional[str]) -> bool:
    """Twilio's scheme: base64 HMAC-SHA1 of the URL plus sorted key+value pairs."""
    if not auth_token or not signature:
        return False
    payload = url + "".join(key + value for key, value in sorted(params))
    expected = base64.b64encode(
        hmac.new(auth_token.encode(), payload.encode(), hashlib.sha1).digest()
    ).decode()
    return _equal(expected, signature)


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


def _secret(name: str, provider: str) -> str:
    """The configured secret, or a 403: an unconfigured webhook accepts nothing."""
    value = os.getenv(name, "")
    if not value:
        logger.warning("%s webhook refused: %s is not set", provider, name)
        raise HTTPException(status_code=403, detail=f"{provider} webhook verification is not configured")
    return value


def _reject(provider: str, what: str) -> HTTPException:
    """A 401 for a request that failed verification, logged without its contents."""
    logger.warning("%s webhook refused: %s", provider, what)
    return HTTPException(status_code=401, detail=f"{provider} webhook verification failed")


async def require_bolna(request: Request) -> None:
    """Bolna: a valid X-Bolna-Signature and the Bearer key, both required."""
    secret = _secret("BOLNA_WEBHOOK_SECRET", "Bolna")
    api_key = _secret("RAG_API_KEY", "Bolna")
    body = await request.body()
    if not verify_hmac_sha256(secret, body, request.headers.get("x-bolna-signature")):
        raise _reject("Bolna", "missing or wrong signature")
    if not verify_bearer(api_key, request.headers.get("authorization")):
        raise _reject("Bolna", "missing or wrong key")


async def require_retell(request: Request) -> None:
    """Retell: a valid, recent X-Retell-Signature keyed by RETELL_API_KEY."""
    api_key = _secret("RETELL_API_KEY", "Retell")
    body = await request.body()
    if not verify_retell(api_key, body, request.headers.get("x-retell-signature")):
        raise _reject("Retell", "missing, stale or wrong signature")


async def require_sarvam(request: Request) -> None:
    """Sarvam: a valid X-Sarvam-Signature keyed by SARVAM_WEBHOOK_SECRET."""
    secret = _secret("SARVAM_WEBHOOK_SECRET", "Sarvam")
    body = await request.body()
    if not verify_hmac_sha256(secret, body, request.headers.get("x-sarvam-signature")):
        raise _reject("Sarvam", "missing or wrong signature")


def twilio_signed_url(request: Request) -> str:
    """The URL Twilio signed: PUBLIC_BASE_URL plus the path and query, else the request URL."""
    base = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    if not base:
        return str(request.url)
    query = request.url.query
    return f"{base}{request.url.path}" + (f"?{query}" if query else "")


async def require_twilio(request: Request) -> None:
    """Twilio: a valid X-Twilio-Signature over the URL and form params."""
    token = _secret("TWILIO_AUTH_TOKEN", "Twilio")
    params = []
    if request.headers.get("content-type", "").startswith("application/x-www-form-urlencoded"):
        form = await request.form()
        params = [(key, str(value)) for key, value in form.multi_items()]
    if not verify_twilio(token, twilio_signed_url(request), params,
                         request.headers.get("x-twilio-signature")):
        raise _reject("Twilio", "missing or wrong signature")
