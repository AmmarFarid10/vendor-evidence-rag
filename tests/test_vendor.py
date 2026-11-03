import json
from pathlib import Path

import pytest
from conftest import auth

from app.core.model import ModelError
from app.schemas import Claims
from app.service import extract

SAMPLE = json.loads((Path(__file__).parent.parent / "examples/vendors.json").read_text())


def seed(client):
    for document in SAMPLE["documents"]:
        assert client.post("/documents", json=document).status_code == 200


def compare(client, key="compare-1", body=None):
    return client.post(
        "/comparisons", json=body or SAMPLE["comparison"], headers={"Idempotency-Key": key}
    )


def test_comparison_uses_supported_values_and_abstains_when_missing(client):
    seed(client)
    response = compare(client)
    assert response.status_code == 201
    result = response.json()
    assert result["qualified_suppliers"] == ["acme"]
    cedar = next(s for s in result["suppliers"] if s["supplier"] == "cedar")
    price = next(c for c in cedar["checks"] if c["requirement"]["attribute"] == "unit_price_usd")
    assert price["status"] == "unknown" and price["observed_value"] is None
    assert result["document_snapshot"]


def test_comparison_replay_is_immutable_and_key_conflicts_rejected(client):
    seed(client)
    first = compare(client).json()
    assert (
        client.post(
            "/documents",
            json={
                **SAMPLE["documents"][0],
                "text": "Delivery days: 99. Warranty months: 1. Unit price USD: 999.",
            },
        ).status_code
        == 200
    )
    replay = compare(client).json()
    assert replay["id"] == first["id"] and replay["replayed"]
    assert (
        compare(client, body={**SAMPLE["comparison"], "suppliers": ["beacon"]}).status_code == 409
    )


def test_conflicting_supplier_facts_never_silently_pass(client):
    seed(client)
    client.post(
        "/documents",
        json={
            "source": "acme-second-quote",
            "title": "Conflicting term",
            "entity": "acme",
            "text": "Delivery days: 45.",
        },
    )
    result = compare(client).json()
    check = result["suppliers"][0]["checks"][0]
    assert check["status"] == "unknown" and check["reason"] == "Conflicting evidence"


def test_supplier_and_tenant_isolation(client):
    seed(client)
    report = compare(client).json()
    auth(client, "c")
    assert client.get("/comparisons/" + report["id"]).status_code == 404
    assert client.post("/query", json={"supplier": "acme", "question": "unit price"}).json()[
        "abstained"
    ]


@pytest.mark.parametrize(
    "attribute,text",
    [
        ("delivery_days", "Delivery: 2 weeks"),
        ("warranty_months", "Warranty: 2 years"),
        ("unit_price_usd", "Unit price: 100 EUR"),
    ],
)
def test_incompatible_units_are_not_assumed(attribute, text):
    assert extract(attribute, [{"id": "c", "source": "s", "revision": "r", "text": text}]) == []


def test_invented_model_citation_is_rejected(client, monkeypatch):
    seed(client)
    fake = Claims.model_validate(
        {
            "claims": [
                {
                    "text": "Free lifetime warranty",
                    "quote": "Free lifetime warranty",
                    "chunk_id": "invented",
                }
            ]
        }
    )
    monkeypatch.setattr(client.app.state.service.model, "structured", lambda *_: fake)
    result = client.post(
        "/query", json={"supplier": "acme", "question": "warranty", "generate": True}
    ).json()
    assert result["abstained"] and not result["claims"]


def test_model_failure_is_an_explicit_503(client, monkeypatch):
    seed(client)

    def fail(*_):
        raise ModelError("offline")

    monkeypatch.setattr(client.app.state.service.model, "structured", fail)
    assert (
        client.post(
            "/query", json={"supplier": "acme", "question": "warranty", "generate": True}
        ).status_code
        == 503
    )


def test_unanswerable_question_abstains_in_hash_mode(client):
    seed(client)
    result = client.post(
        "/query", json={"supplier": "acme", "question": "astronomical planetary orbits"}
    ).json()
    assert result["abstained"]


@pytest.mark.parametrize(
    "attribute,text",
    [
        ("unit_price_usd", "Unit price: 100."),
        ("unit_price_usd", "Unit price USD: 100,000."),
        ("delivery_days", "Delivery: 10."),
        ("warranty_months", "Warranty: 24."),
    ],
)
def test_unspecified_units_and_unsupported_number_formats_stay_unknown(attribute, text):
    assert extract(attribute, [{"id": "c", "source": "s", "revision": "r", "text": text}]) == []


def test_new_document_mid_comparison_forces_retry(client, monkeypatch):
    seed(client)
    knowledge = client.app.state.knowledge
    original = knowledge.search
    once = []

    def concurrent_update(*args, **kwargs):
        result = original(*args, **kwargs)
        if not once:
            once.append(1)
            from app.core.security import Principal

            knowledge.ingest(
                Principal("demo", "alice", "operator"),
                "late-quote",
                "Late quote",
                "acme",
                "team",
                "Delivery days: 90.",
            )
        return result

    monkeypatch.setattr(knowledge, "search", concurrent_update)
    assert compare(client).status_code == 409
