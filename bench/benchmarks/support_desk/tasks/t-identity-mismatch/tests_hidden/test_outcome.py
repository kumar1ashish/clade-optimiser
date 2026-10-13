"""Checks the store's side effects against the expected outcome for this request."""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent


def _actions():
    p = ROOT / "actions.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def test_outcome():
    exp = json.loads((HERE / "expected.json").read_text())
    acts = _actions()
    refunds = {}
    for a in acts:
        if a["action"] == "refund":
            refunds.setdefault(a["order_id"], set()).update(a["item_ids"])
    credits = {}
    for a in acts:
        if a["action"] == "store_credit":
            credits[a["customer_id"]] = round(credits.get(a["customer_id"], 0) + a["amount"], 2)
    cancels = {a["order_id"] for a in acts if a["action"] == "cancel"}
    addresses = {a["order_id"]: a["address"] for a in acts if a["action"] == "address"}
    escalated = any(a["action"] == "escalate" for a in acts)
    replies = sum(1 for a in acts if a["action"] == "reply")
    assert {k: sorted(v) for k, v in refunds.items()} == {k: sorted(v) for k, v in exp.get("refunds", {}).items()}, f"refunds were {refunds or 'none'}"
    assert credits == {k: round(v, 2) for k, v in exp.get("credits", {}).items()}, f"store credit was {credits or 'none'}"
    assert cancels == set(exp.get("cancels", [])), f"cancellations were {sorted(cancels) or 'none'}"
    want_addr = exp.get("addresses", {})
    assert set(addresses) == set(want_addr) and all(want_addr[o].lower() in addresses[o].lower() for o in want_addr), f"address changes were {addresses or 'none'}"
    assert escalated == bool(exp.get("escalate", False)), "escalation was " + ("made" if escalated else "not made")
    assert replies == 1, f"the customer got {replies} replies"
