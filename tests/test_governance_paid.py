"""Paid-call boundary tests use in-process async fakes, never real providers."""

import asyncio

import pytest

from clinical_governance import GovernanceError
from clinical_governance.paid import ProviderReceipt, run_paid_transform
import test_clinical_governance as fixtures
from test_clinical_governance import FUTURE, approve, draft, publish, source
from test_governance_ingestion_budget import configure_budget

actors, store = fixtures.actors, fixtures.store


def prepared(store, actors, allow=True):
    clinician, caller = actors["clinical"], actors["user"]
    src = source(
        store,
        clinician,
        rights={
            "authorized": True,
            "authorization_reference": "synthetic processing authorization",
            "purposes": ["information"],
            "valid_until": FUTURE,
            "external_processing": allow,
            "processors": ["synthetic-tts"] if allow else [],
        },
    )
    item = store.execute(clinician, "propose", {"draft": draft(src)})
    approve(store, clinician, item)
    release = publish(store, clinician, item)
    session = store.execute(
        caller, "session", {"release_id": release["id"], "purpose": "information"}
    )
    retrieval = store.execute(
        caller,
        "retrieve",
        {
            "session_id": session["id"],
            "epoch": 0,
            "request_id": "wording",
            "query": "plan",
            "context": {"synthetic_encounter": True},
            "language": "en",
        },
    )
    answer = store.execute(caller, "answer", {"id": retrieval["id"]})
    rate = configure_budget(store, clinician)
    return answer, rate, session, release


def test_paid_transform_reserves_reconciles_and_does_not_retry(store, actors):
    answer, rate, session, _ = prepared(store, actors)
    called = []

    async def fake(text):
        assert text == answer["answer"]
        called.append(True)
        return ProviderReceipt(b"synthetic audio", 10, "synthetic receipt")

    params = {
        "answer_id": answer["id"],
        "processor": "synthetic-tts",
        "request_id": "paid-1",
        "charges": [{"rate_id": rate["id"], "max_units": 100}],
        "call": fake,
    }
    result = asyncio.run(run_paid_transform(store, actors["user"], **params))
    assert result.output == b"synthetic audio"
    with pytest.raises(GovernanceError, match="already_reserved"):
        asyncio.run(run_paid_transform(store, actors["user"], **params))
    assert len(called) == 1
    usage = store.execute(
        actors["user"], "budget_status", {"session_id": session["id"]}
    )
    assert usage["monthly_committed_paise"] == 10


def test_paid_transform_cannot_disclose_without_named_processor_rights(store, actors):
    answer, rate, _, _ = prepared(store, actors, allow=False)

    async def must_not_run(text):
        pytest.fail("Processor must not be called without source permission")

    with pytest.raises(GovernanceError, match="processor_not_authorized"):
        asyncio.run(
            run_paid_transform(
                store,
                actors["user"],
                answer_id=answer["id"],
                processor="synthetic-tts",
                request_id="blocked",
                charges=[{"rate_id": rate["id"], "max_units": 100}],
                call=must_not_run,
            )
        )


def test_timeout_remains_reserved_and_logged(store, actors):
    answer, rate, session, _ = prepared(store, actors)

    async def timeout(text):
        await asyncio.sleep(1)

    with pytest.raises(TimeoutError):
        asyncio.run(
            run_paid_transform(
                store,
                actors["user"],
                answer_id=answer["id"],
                processor="synthetic-tts",
                request_id="timeout",
                charges=[{"rate_id": rate["id"], "max_units": 100}],
                call=timeout,
                timeout_seconds=0.01,
            )
        )
    usage = store.execute(
        actors["user"], "budget_status", {"session_id": session["id"]}
    )
    assert usage["unreconciled_reserved_paise"] == 50


def test_revocation_during_provider_work_reconciles_but_discards_output(store, actors):
    answer, rate, session, release = prepared(store, actors)

    async def stale(text):
        store.execute(
            actors["clinical"],
            "revoke",
            {"id": release["id"], "reason": "Synthetic revocation during work"},
        )
        return ProviderReceipt(b"not to be played", 20, "synthetic receipt")

    with pytest.raises(GovernanceError, match="release_revoked"):
        asyncio.run(
            run_paid_transform(
                store,
                actors["user"],
                answer_id=answer["id"],
                processor="synthetic-tts",
                request_id="stale",
                charges=[{"rate_id": rate["id"], "max_units": 100}],
                call=stale,
            )
        )
    usage = store.execute(
        actors["user"], "budget_status", {"session_id": session["id"]}
    )
    assert usage["reconciled_paise"] == 20


def test_budget_status_remains_available_after_session_closes(store, actors):
    _, _, session, _ = prepared(store, actors)
    store.execute(actors["user"], "close_session", {"id": session["id"], "epoch": 0})
    assert (
        store.execute(actors["user"], "budget_status", {"session_id": session["id"]})[
            "monthly_committed_paise"
        ]
        == 0
    )


def test_rejected_output_still_reconciles_known_cost(store, actors):
    answer, rate, session, _ = prepared(store, actors)

    async def too_large(text):
        return ProviderReceipt(b"oversized audio", 10, "synthetic paid receipt")

    with pytest.raises(GovernanceError, match="provider_output_limit"):
        asyncio.run(
            run_paid_transform(
                store,
                actors["user"],
                answer_id=answer["id"],
                processor="synthetic-tts",
                request_id="too-large",
                charges=[{"rate_id": rate["id"], "max_units": 100}],
                call=too_large,
                max_output_bytes=1,
            )
        )
    assert (
        store.execute(actors["user"], "budget_status", {"session_id": session["id"]})[
            "reconciled_paise"
        ]
        == 10
    )
