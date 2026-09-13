"""INR ledger with conservative reservations and explicit reconciliation.

All amounts are integer paise, not binary floating point rupees. No provider
prices are bundled. Unpriced or uncertain operations cannot bypass limits.
"""

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING

from .models import GovernanceError, timestamp, utcnow
from .store import require


def positive_decimal(value):
    require(type(value) in (int, str), "decimal_string_or_integer_required", 422)
    try:
        result = Decimal(value)
        require(result.is_finite() and 0 < result <= Decimal("1e12"), "positive_bounded_decimal_required", 422)
        return result
    except InvalidOperation:
        raise GovernanceError("invalid_decimal", 422) from None


def _amount(value):
    require(type(value) is int and 0 <= value <= 10**12, "integer_paise_required", 422)
    return value


def _policy(store, db, actor):
    try:
        return store._get(db, actor, "budget_policy", "budget_policy")
    except GovernanceError as exc:
        if exc.http_status == 404:
            raise GovernanceError("inr_budget_not_configured") from None
        raise


def _usage(store, db, actor, period, session_id):
    reservations = store._list(db, actor, "reservation")
    settlements = {row["reservation_id"]: row for row in store._list(db, actor, "settlement")}
    monthly = session = reserved = settled = 0
    for reservation in reservations:
        settlement = settlements.get(reservation["id"])
        amount = settlement["actual_paise"] if settlement else reservation["reserved_paise"]
        if reservation["period"] == period:
            monthly += amount
            if settlement:
                settled += amount
            else:
                reserved += amount
        if reservation["session_id"] == session_id:
            session += amount
    return {"monthly_committed_paise": monthly, "session_committed_paise": session,
            "unreconciled_reserved_paise": reserved, "reconciled_paise": settled,
            "period": period, "period_basis": "UTC_reservation_month"}


def dispatch_budget(store, db, actor, operation, data):
    if operation == "budget_configure":
        actor.require("operator")
        try:
            old = _policy(store, db, actor)
        except GovernanceError as exc:
            if exc.code != "inr_budget_not_configured":
                raise
            old = {"revision": 0}
        require(data["expected_revision"] == old["revision"], "budget_revision_conflict")
        payload = {"session_limit_paise": _amount(data["session_limit_paise"]),
                   "monthly_limit_paise": _amount(data["monthly_limit_paise"]),
                   "configured_by": actor.id, "at": utcnow()}
        result = store._put(db, actor, "budget_policy", payload, "budget_policy")
        return result, {"revision": result["revision"], **payload}
    if operation == "budget_rate":
        actor.require("operator")
        require(isinstance(data["service"], str) and 0 < len(data["service"]) <= 100, "service_required", 422)
        require(isinstance(data["version"], str) and bool(data["version"]) and bool(data.get("billing_reference")), "rate_provenance_required", 422)
        require(data["unit"] in ("token", "second", "character", "request", "minute", "byte"), "invalid_billing_unit", 422)
        require(timestamp(data["valid_until"]) > datetime.now(timezone.utc), "rate_expired")
        price = positive_decimal(data["paise_per_unit"])
        result = store._put(db, actor, "rate", {"service": data["service"], "version": data["version"],
            "paise_per_unit": str(price), "unit": data["unit"], "currency": "INR", "valid_until": data["valid_until"],
            "billing_reference": data["billing_reference"][:500], "exchange_basis": data.get("exchange_basis", "native_INR"), "at": utcnow()})
        return result, {"rate_id": result["id"], "rate_hash": result["hash"], "service": result["service"], "version": result["version"], "unit": result["unit"]}
    if operation == "budget_reserve":
        actor.require("requester", "clinician", "billing")
        session = store._session(db, actor, data["session_id"])
        require(session["owner"] == actor.id or "billing" in actor.roles, "session_owner_required", 403)
        store._release(db, actor, session["release_id"], session["purpose"])
        require(data["epoch"] == session["epoch"], "stale_epoch")
        policy = _policy(store, db, actor)
        key = data["request_id"]
        require(isinstance(key, str) and 0 < len(key) <= 100, "reservation_id_required", 422)
        id_ = "reservation_" + store.fingerprint([actor.tenant, actor.id, session["id"], key])[:32]
        fingerprint = store.fingerprint(data)
        try:
            previous = store._get(db, actor, id_, "reservation")
        except GovernanceError as exc:
            if exc.http_status != 404:
                raise
        else:
            require(previous["request_fingerprint"] == fingerprint, "reservation_idempotency_conflict")
            return {**previous, "replayed": True}, {"reservation_id": id_, "replayed": True}
        require(isinstance(data["charges"], list) and 0 < len(data["charges"]) <= 20, "bounded_charge_list_required", 422)
        charges, total = [], 0
        for charge in data["charges"]:
            rate = store._get(db, actor, charge["rate_id"], "rate")
            require(timestamp(rate["valid_until"]) > datetime.now(timezone.utc), "rate_expired")
            units = positive_decimal(charge["max_units"])
            amount = int((positive_decimal(rate["paise_per_unit"]) * units).to_integral_value(rounding=ROUND_CEILING))
            total += amount
            charges.append({"rate_id": rate["id"], "rate_hash": rate["hash"], "max_units": str(units),
                            "unit": rate["unit"], "service": rate["service"], "reserved_paise": amount})
        period = datetime.now(timezone.utc).strftime("%Y-%m")
        usage = _usage(store, db, actor, period, session["id"])
        require(usage["monthly_committed_paise"] + total <= policy["monthly_limit_paise"], "monthly_inr_budget_exhausted")
        require(usage["session_committed_paise"] + total <= policy["session_limit_paise"], "session_inr_budget_exhausted")
        result = store._put(db, actor, "reservation", {"session_id": session["id"], "epoch": session["epoch"],
            "owner": actor.id, "charges": charges, "reserved_paise": total, "period": period,
            "budget_policy_revision": policy["revision"], "request_fingerprint": fingerprint, "at": utcnow()}, id_)
        return result, {"reservation_id": id_, "session_id": session["id"], "charges": charges, "reserved_paise": total, "period": period}
    if operation == "budget_reconcile":
        actor.require("billing", "operator")
        reservation = store._get(db, actor, data["id"], "reservation")
        actual = _amount(data["actual_paise"])
        require(data.get("basis") in ("provider_usage", "provider_invoice", "confirmed_no_charge"), "billing_basis_required", 422)
        require(bool(data.get("reference")), "billing_reference_required", 422)
        require(data["basis"] != "confirmed_no_charge" or actual == 0, "no_charge_must_be_zero", 422)
        id_ = "settlement_" + reservation["id"]
        try:
            previous = store._get(db, actor, id_, "settlement")
        except GovernanceError as exc:
            if exc.http_status != 404:
                raise
            previous = {"revision": 0}
        if data.get("expected_revision") != previous["revision"]:
            # Exact retry is safe; a different correction needs an explicit revision.
            if previous.get("actual_paise") == actual and previous.get("reference_fingerprint") == store.fingerprint(data["reference"]):
                return previous, {"settlement_id": id_, "replayed": True}
            raise GovernanceError("settlement_revision_conflict")
        result = store._put(db, actor, "settlement", {"reservation_id": reservation["id"], "actual_paise": actual,
            "basis": data["basis"], "reference_fingerprint": store.fingerprint(data["reference"]),
            "over_reserved_amount": actual > reservation["reserved_paise"], "at": utcnow()}, id_)
        # An overrun is an observed liability, never discarded to make a cap appear true.
        return result, {"settlement_id": id_, "reservation_id": reservation["id"], "actual_paise": actual,
                        "basis": result["basis"], "over_reserved_amount": result["over_reserved_amount"]}
    if operation == "budget_status":
        actor.require("operator", "billing", "requester", "clinician")
        session = store._session(db, actor, data["session_id"])
        policy = _policy(store, db, actor)
        period = datetime.now(timezone.utc).strftime("%Y-%m")
        usage = _usage(store, db, actor, period, session["id"])
        return {"policy": policy, **usage}, {"session_id": session["id"], "period": period}
    raise GovernanceError("unknown_budget_operation", 404)
