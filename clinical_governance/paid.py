"""Server-side paid transform boundary; no provider is enabled by this module.

An application adapter supplies the callable and bounded quote. Clients cannot
send a callable or reconcile a bill through this helper. It is suitable for
transforming approved text (e.g. TTS), not authorizing autonomous clinical acts.
"""

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from .models import Actor, GovernanceError, utcnow
from .store import require


@dataclass
class ProviderReceipt:
    output: bytes
    actual_paise: Optional[int]
    reference: str


def dispatch_provider(store, db, actor, operation, data):
    actor.require("provider")
    if operation == "provider_authorize":
        answer = store._get(db, actor, data["answer_id"], "answer")
        retrieval = store._get(db, actor, answer["retrieval_id"], "retrieval")
        require(retrieval["owner"] == actor.id, "request_owner_required", 403)
        store._revalidate(db, actor, retrieval)
        require(
            store._policy(db, actor)["revision"] == retrieval["policy_revision"],
            "review_policy_changed_retrieve_again",
        )
        require(
            store._review_status(db, actor, retrieval) in ("approved", "not_required"),
            "request_review_not_approved",
        )
        require(
            isinstance(data["processor"], str) and 0 < len(data["processor"]) <= 100,
            "processor_required",
            422,
        )
        for claim in answer["claims"]:
            for anchor in claim["evidence"]:
                source = store._get(db, actor, anchor["source_id"], "source")
                require(
                    source["rights"]["external_processing"]
                    and data["processor"] in source["rights"].get("processors", []),
                    "processor_not_authorized",
                    403,
                )
        return {
            "authorized": True,
            "session_id": retrieval["session_id"],
            "epoch": retrieval["epoch"],
            "approved_text": answer["answer"],
        }, {
            "answer_id": answer["id"],
            "processor": data["processor"],
            "release_id": retrieval["release_id"],
            "input_sequence": retrieval["input_sequence"],
            "clinical_action_authorized": False,
        }
    if operation == "provider_result":
        reservation = store._get(db, actor, data["reservation_id"], "reservation")
        require(reservation["owner"] == actor.id, "reservation_owner_required", 403)
        require(
            data["status"]
            in (
                "completed",
                "failed_or_cancelled",
                "stale_discarded",
                "output_rejected",
            ),
            "invalid_provider_status",
            422,
        )
        result = store._put(
            db,
            actor,
            "provider_result",
            {
                "reservation_id": reservation["id"],
                "answer_id": data["answer_id"],
                "processor": data["processor"],
                "status": data["status"],
                "at": utcnow(),
                "delivery": "not_confirmed",
            },
        )
        return result, {
            "result_id": result["id"],
            **{
                key: result[key]
                for key in (
                    "reservation_id",
                    "answer_id",
                    "processor",
                    "status",
                    "delivery",
                )
            },
        }
    raise GovernanceError("unknown_provider_operation", 404)


async def run_paid_transform(
    store,
    actor: Actor,
    *,
    answer_id: str,
    processor: str,
    request_id: str,
    charges: list[dict],
    call: Callable[[str], Awaitable[ProviderReceipt]],
    timeout_seconds: float = 30,
    max_output_bytes: int = 20 * 1024 * 1024,
) -> ProviderReceipt:
    """Reserve before calling; reconcile even if the completed output is stale.

    The caller is trusted server code, not a user-supplied plugin. The adapter
    must enforce its quoted maximum units, timeouts, retries and output bounds.
    Unknown failures stay reserved and are never automatically retried.
    """
    actor.require("requester", "clinician")
    require(
        0 < timeout_seconds <= 120
        and type(max_output_bytes) is int
        and 0 < max_output_bytes <= 20 * 1024 * 1024,
        "invalid_provider_bounds",
        422,
    )
    service_actor = Actor(
        actor.id,
        actor.tenant,
        actor.roles | {"provider", "billing"},
        actor.purposes,
        actor.clinical_authority,
    )
    authorization = await asyncio.to_thread(
        store.execute,
        service_actor,
        "provider_authorize",
        {"answer_id": answer_id, "processor": processor},
    )
    reservation = await asyncio.to_thread(
        store.execute,
        service_actor,
        "budget_reserve",
        {
            "session_id": authorization["session_id"],
            "epoch": authorization["epoch"],
            "request_id": request_id,
            "charges": charges,
        },
    )
    require(
        not reservation.get("replayed"),
        "paid_operation_already_reserved_no_automatic_retry",
    )
    try:
        # Recheck after reservation and immediately before external disclosure.
        await asyncio.to_thread(
            store.execute,
            service_actor,
            "provider_authorize",
            {"answer_id": answer_id, "processor": processor},
        )
        receipt = await asyncio.wait_for(
            call(authorization["approved_text"]), timeout=timeout_seconds
        )
        require(isinstance(receipt, ProviderReceipt), "invalid_provider_receipt")
        require(
            receipt.actual_paise is None
            or (
                type(receipt.actual_paise) is int
                and 0 <= receipt.actual_paise <= 10**12
                and bool(receipt.reference)
            ),
            "invalid_provider_receipt",
        )
    except BaseException:
        await asyncio.shield(
            asyncio.to_thread(
                store.execute,
                service_actor,
                "provider_result",
                {
                    "reservation_id": reservation["id"],
                    "answer_id": answer_id,
                    "processor": processor,
                    "status": "failed_or_cancelled",
                },
            )
        )
        raise
    if receipt.actual_paise is not None:
        await asyncio.to_thread(
            store.execute,
            service_actor,
            "budget_reconcile",
            {
                "id": reservation["id"],
                "actual_paise": receipt.actual_paise,
                "basis": "provider_usage",
                "reference": receipt.reference,
                "expected_revision": 0,
            },
        )
    try:
        require(
            isinstance(receipt.output, bytes)
            and len(receipt.output) <= max_output_bytes,
            "provider_output_limit",
        )
        await asyncio.to_thread(
            store.execute,
            service_actor,
            "provider_authorize",
            {"answer_id": answer_id, "processor": processor},
        )
    except GovernanceError as error:
        await asyncio.to_thread(
            store.execute,
            service_actor,
            "provider_result",
            {
                "reservation_id": reservation["id"],
                "answer_id": answer_id,
                "processor": processor,
                "status": "output_rejected"
                if error.code == "provider_output_limit"
                else "stale_discarded",
            },
        )
        raise
    await asyncio.to_thread(
        store.execute,
        service_actor,
        "provider_result",
        {
            "reservation_id": reservation["id"],
            "answer_id": answer_id,
            "processor": processor,
            "status": "completed",
        },
    )
    return receipt
