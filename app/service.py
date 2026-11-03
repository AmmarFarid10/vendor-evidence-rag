"""Evidence extraction is separated from deterministic purchasing requirements."""

import json
import re
from decimal import Decimal
from uuid import uuid4

from app.core.database import digest
from app.core.model import LocalModel
from app.core.retrieval import Knowledge, validate_quotes
from app.core.security import Principal
from app.schemas import Claims, Comparison, Query

PATTERNS = {
    "delivery_days": r"(?:delivery(?: lead time)?|lead time)(?: days)?\s*:\s*(\d+(?:\.\d+)?)\s*(?:days)?",
    "warranty_months": r"warranty(?: months)?\s*:\s*(\d+(?:\.\d+)?)\s*(?:months)?",
    "unit_price_usd": r"unit price(?:\s+USD\s*:\s*|\s*:\s*(?:USD\s*|\$))(\d+(?:\.\d+)?)(?!\d|[.,]\d)",
    "capacity_units": r"capacity(?: units)?\s*:\s*(\d+(?:\.\d+)?)\s*(?:units)?",
}
QUESTIONS = {
    "delivery_days": "delivery lead time shipment days",
    "warranty_months": "warranty months guarantee",
    "unit_price_usd": "unit price USD cost",
    "capacity_units": "capacity units production",
}


def extract(attribute: str, evidence: list[dict]) -> list[dict]:
    findings = []
    for chunk in evidence:
        for match in re.finditer(PATTERNS[attribute], chunk["text"], re.I):
            # Reject an explicitly conflicting unit/currency instead of inventing a conversion.
            rest = chunk["text"][match.end() : match.end() + 12].strip().lower()
            quote = match.group(0)
            if attribute == "delivery_days" and "day" not in quote.lower():
                continue
            if attribute == "warranty_months" and "month" not in quote.lower():
                continue
            if attribute == "capacity_units" and "units" not in quote.lower():
                continue
            if attribute == "unit_price_usd" and re.match(r"^(eur|gbp|pkr|cad)\b", rest):
                continue
            if attribute == "delivery_days" and re.match(r"^(weeks|months|hours)\b", rest):
                continue
            if attribute == "warranty_months" and re.match(r"^(years|days|weeks)\b", rest):
                continue
            findings.append(
                {
                    "value": str(Decimal(match.group(1))),
                    "text": quote,
                    "quote": quote,
                    "chunk_id": chunk["id"],
                    "source": chunk["source"],
                    "revision": chunk["revision"],
                }
            )
    return findings


class Procurement:
    def __init__(self, knowledge: Knowledge, model: LocalModel | None = None):
        self.knowledge, self.db = knowledge, knowledge.db
        self.model = model or LocalModel()
        with self.db.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS reports(tenant TEXT,id TEXT,creator TEXT,request_key TEXT,"
                "input_hash TEXT,result TEXT,PRIMARY KEY(tenant,id),UNIQUE(tenant,creator,request_key))"
            )

    def compare(self, principal: Principal, body: Comparison, key: str) -> dict:
        payload = body.model_dump(mode="json")
        with self.db.connect() as db:
            row = db.execute(
                "SELECT * FROM reports WHERE tenant=? AND creator=? AND request_key=?",
                (principal.tenant, principal.subject, key),
            ).fetchone()
            if row:
                if row["input_hash"] != digest(payload):
                    raise ValueError("Idempotency key already used for another comparison")
                return {**json.loads(row["result"]), "replayed": True}
        suppliers = []
        snapshot = self.knowledge.snapshot(principal, body.suppliers)
        for supplier in body.suppliers:
            checks = []
            for requirement in body.requirements:
                evidence = self.knowledge.search(
                    principal, QUESTIONS[requirement.attribute], [supplier], limit=12
                )
                findings = extract(requirement.attribute, evidence)
                if body.extractor == "ollama" and findings:
                    selected = self.model.structured(
                        Claims,
                        "Select exact evidence quotes about this requirement. Do not paraphrase. "
                        "Only return text that is copied from its quote and exact chunk IDs.",
                        {"requirement": requirement.model_dump(mode="json"), "evidence": evidence},
                    )
                    claims = [c.model_dump() for c in selected.claims]
                    if not validate_quotes(claims, evidence):
                        findings = []
                    else:
                        # Model output cannot introduce a numeric value. Parse and compare locally.
                        accepted = {(c["chunk_id"], c["quote"]) for c in claims}
                        findings = [
                            f
                            for f in findings
                            if any(
                                cid == f["chunk_id"] and f["quote"] in quote
                                for cid, quote in accepted
                            )
                        ]
                values = {Decimal(f["value"]) for f in findings}
                value = next(iter(values)) if len(values) == 1 else None
                status = "unknown"
                if value is not None:
                    met = (
                        value <= requirement.value
                        if requirement.operator == "le"
                        else value >= requirement.value
                    )
                    status = "met" if met else "not_met"
                checks.append(
                    {
                        "requirement": requirement.model_dump(mode="json"),
                        "status": status,
                        "observed_value": str(value) if value is not None else None,
                        "reason": "Conflicting evidence"
                        if len(values) > 1
                        else "No supported value"
                        if not findings
                        else "Verified numeric comparison",
                        "citations": findings,
                    }
                )
            suppliers.append(
                {
                    "supplier": supplier,
                    "checks": checks,
                    "all_requirements_met": all(c["status"] == "met" for c in checks),
                }
            )
        result: dict = {
            "id": str(uuid4()),
            "suppliers": suppliers,
            "qualified_suppliers": [s["supplier"] for s in suppliers if s["all_requirements_met"]],
            "document_snapshot": snapshot,
            "embedding_model": self.knowledge.encoder.name,
            "extractor": body.extractor,
            "replayed": False,
        }
        with self.db.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # A concurrent duplicate receives the first immutable result, not another report.
            row = db.execute(
                "SELECT input_hash,result FROM reports WHERE tenant=? AND creator=? AND request_key=?",
                (principal.tenant, principal.subject, key),
            ).fetchone()
            if row:
                if row["input_hash"] != digest(payload):
                    raise ValueError("Idempotency key conflict")
                return {**json.loads(row["result"]), "replayed": True}
            if self.knowledge.snapshot(principal, body.suppliers, db) != snapshot:
                raise ValueError("Source documents changed during comparison; retry with a new key")
            db.execute(
                "INSERT INTO reports VALUES(?,?,?,?,?,?)",
                (
                    principal.tenant,
                    result["id"],
                    principal.subject,
                    key,
                    digest(payload),
                    json.dumps(result),
                ),
            )
            self.db.audit(
                db,
                principal.tenant,
                principal.subject,
                "comparison_created",
                result["id"],
                {"suppliers": len(suppliers), "requirements": len(body.requirements)},
            )
        return result

    def query(self, principal: Principal, body: Query) -> dict:
        evidence = self.knowledge.search(principal, body.question, [body.supplier], mode=body.mode)
        if not evidence:
            return {"abstained": True, "claims": [], "evidence": []}
        claims = [{"text": e["text"], "quote": e["text"], "chunk_id": e["id"]} for e in evidence]
        if body.generate:
            response = self.model.structured(
                Claims,
                "Answer by selecting relevant exact quotes from evidence. Never paraphrase or invent facts. "
                "Return no claims if the question is not supported.",
                {"question": body.question, "evidence": evidence},
            )
            claims = [c.model_dump() for c in response.claims]
        valid = validate_quotes(claims, evidence)
        return {
            "abstained": not valid,
            "claims": claims if valid else [],
            "evidence": evidence,
            "mode": "validated-local-model" if body.generate else "cited-extracts",
        }
