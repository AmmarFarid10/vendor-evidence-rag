"""Exercise the real FastAPI API against temporary storage; never send external messages."""

import argparse
import json
import os
import secrets
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient

from app.api import app

ROOT = Path(__file__).resolve().parents[1]


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--semantic", action="store_true")
    args = parser.parse_args()
    operator, reviewer = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    os.environ["AUTH_KEYS_JSON"] = json.dumps(
        [
            {"tenant": "demo", "subject": "alice", "role": "operator", "key": operator},
            {"tenant": "demo", "subject": "bob", "role": "reviewer", "key": reviewer},
        ]
    )
    os.environ["EMBEDDER"] = "fastembed" if args.semantic else "hash"
    with tempfile.TemporaryDirectory() as temp:
        os.environ["DATA_PATH"] = str(Path(temp) / "demo.sqlite3")
        with TestClient(app) as client:
            client.headers["Authorization"] = "Bearer " + operator
            summary = exercise(client, operator, reviewer)
    print(json.dumps(summary, indent=2))
    return summary


def exercise(client, operator, reviewer):
    sample = json.loads((ROOT / "examples/vendors.json").read_text())
    for document in sample["documents"]:
        assert client.post("/documents", json=document).status_code == 200
    response = client.post(
        "/comparisons", json=sample["comparison"], headers={"Idempotency-Key": "demo-comparison"}
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["qualified_suppliers"] == sample["expected_qualified"]
    assert client.post(
        "/comparisons", json=sample["comparison"], headers={"Idempotency-Key": "demo-comparison"}
    ).json()["replayed"]
    return {
        "scenario": "Industrial switch RFQ",
        "qualified_suppliers": result["qualified_suppliers"],
        "suppliers": [
            {"supplier": s["supplier"], "checks": [c["status"] for c in s["checks"]]}
            for s in result["suppliers"]
        ],
        "embedding_model": result["embedding_model"],
        "external_llm_called": False,
    }


if __name__ == "__main__":
    main()
