"""Validates the support-desk test split: an independent reading of the policy must agree with every task's expected
outcome, and every hidden test must pass correct behaviour and fail common wrong behaviour.

    python -m bench.benchmarks.support_desk.validate
"""
import datetime as dt
import importlib.util
import json
import re
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "tasks"

# What each email asks for, read by hand: (kind, order, requested item names or new address)
ASK = {
 "address-change-processing": ("address", "O1008", "48 Birch Lane"),
 "address-change-shipped": ("address", "O1009", "300 Pine St"),
 "already-refunded": ("refund", "O1018", ["Bike light"]),
 "cancel-shipped": ("cancel", "O1021", None),
 "damaged-opened-refund": ("refund", "O1004", ["Trekking poles"]),
 "damaged-past-window": ("refund", "O1014", ["Lantern"]),
 "exactly-30-days": ("refund", "O1015", ["Hammock"]),
 "final-sale-damaged": ("refund", "O1005", ["Clearance rain jacket"]),
 "find-right-order": ("refund", "O1020", ["Ultralight tent"]),
 "identity-mismatch": ("refund", "O1010", ["Gaiters"]),
 "large-refund-escalate": ("refund", "O1006", ["Two-person tent", "Sleeping bag"]),
 "large-request-small-eligible": ("refund", "O1017", ["Backpack", "Hiking boots"]),
 "opened-not-damaged": ("refund", "O1003", ["Headlamp"]),
 "order-of-other-customer": ("refund", "O1011", ["Compass"]),
 "partial-final-sale": ("refund", "O1013", ["Merino socks", "Clearance cap"]),
 "partial-opened-and-unopened": ("refund", "O1012", ["Gloves", "Beanie"]),
 "past-window-store-credit": ("refund", "O1002", ["Camping stove"]),
 "processing-cancel": ("refund", "O1007", ["Kayak paddle"]),
 "refund-unopened-in-window": ("refund", "O1001", ["Trail running shoes"]),
 "thirty-one-days": ("refund", "O1016", ["Hammock"]),
}

def oracle(task, db, sender):
    kind, oid, what = ASK[task]
    o = db["orders"][oid]; cust = db["customers"].get(o["customer_id"], {})
    out = {}
    if cust.get("email", "").lower() != sender.lower():
        return out, "identity"
    if kind == "address":
        return ({"addresses": {oid: what}} if o["status"] == "processing" else {}), "address"
    if kind == "cancel" or (kind == "refund" and o["status"] == "processing"):
        return ({"cancels": [oid]} if o["status"] == "processing" else {}), "status"
    if o["status"] != "delivered":
        return {}, "not delivered"
    today = dt.date.fromisoformat(db["today"]); day = (today - dt.date.fromisoformat(o["delivered_date"])).days
    refund, credit = [], 0.0
    for it in o["items"]:
        if it["name"] not in what or it.get("refunded") or it.get("final_sale"):
            continue
        if it.get("opened") and not it.get("damaged"):
            continue
        if day <= 30: refund.append(it)
        else: credit += it["price"]
    if sum(i["price"] for i in refund) + credit > 300:
        return {"escalate": True}, f"day {day}, large"
    if refund: out["refunds"] = {oid: sorted(i["id"] for i in refund)}
    if credit: out["credits"] = {o["customer_id"]: credit}
    return out, f"day {day}"

def actions_for(exp):
    acts = []
    for oid, ids in exp.get("refunds", {}).items(): acts.append({"action": "refund", "order_id": oid, "item_ids": ids})
    for cid, amt in exp.get("credits", {}).items(): acts.append({"action": "store_credit", "customer_id": cid, "amount": amt})
    for oid in exp.get("cancels", []): acts.append({"action": "cancel", "order_id": oid})
    for oid, a in exp.get("addresses", {}).items(): acts.append({"action": "address", "order_id": oid, "address": a + ", Seattle, WA"})
    if exp.get("escalate"): acts.append({"action": "escalate", "reason": "x"})
    return acts

def run_test(tdir, acts):
    """Runs the task's hidden test in-process against a recorded list of store actions."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); shutil.copytree(tdir / "tests_hidden", d / "tests_hidden"); shutil.copytree(tdir / "workspace", d, dirs_exist_ok=True)
        (d / "actions.jsonl").write_text("".join(json.dumps(a) + "\n" for a in acts))
        spec = importlib.util.spec_from_file_location("exam_outcome", d / "tests_hidden" / "test_outcome.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        try:
            mod.test_outcome()
            return True
        except AssertionError:
            return False

def naive(db, task):
    """A careless agent: does literally what the customer asked, no policy."""
    kind, oid, what = ASK[task]; o = db["orders"][oid]
    if kind == "address": return [{"action": "address", "order_id": oid, "address": what}]
    if kind == "cancel": return [{"action": "cancel", "order_id": oid}]
    ids = sorted(i["id"] for i in o["items"] if i["name"] in what)
    return [{"action": "refund", "order_id": oid, "item_ids": ids}]


def check() -> tuple[list, list]:
    """Returns (rows, problems). A problem is a disagreement with the policy or a test that misjudges an outcome."""
    reply = {"action": "reply", "message": "done"}
    rows, problems = [], []
    norm = lambda e: json.dumps({k: v for k, v in sorted(e.items()) if v}, sort_keys=True)
    for task in sorted(ASK):
        tdir = ROOT / task; tj = json.loads((tdir / "task.json").read_text())
        db = json.loads((tdir / "workspace/db.json").read_text()); exp = json.loads((tdir / "tests_hidden/expected.json").read_text())
        sender = re.search(r"From:\s*(\S+)", tj["prompt"]).group(1)
        want, why = oracle(task, db, sender)
        good, careless = actions_for(exp), naive(db, task)
        checks = {
            "passes correct behaviour": run_test(tdir, good + [reply]),
            "fails with no reply": not run_test(tdir, good),
            "fails with two replies": not run_test(tdir, good + [reply, reply]),
            "fails unwanted store credit": not run_test(tdir, good + [{"action": "store_credit", "customer_id": "C100", "amount": 5.0}, reply]),
            "fails doing nothing": None if norm(exp) == "{}" else not run_test(tdir, [reply]),
            # control tasks: the right outcome is exactly what the customer asked for
            "fails the careless agent": None if run_test(tdir, good + [reply]) and norm(oracle_of(careless)) == norm(exp) else not run_test(tdir, careless + [reply]),
        }
        rows.append({"task": task, "basis": why, "agrees": norm(want) == norm(exp), "checks": checks,
                     "test": tj["meta"].get("split") == "test"})
        if norm(want) != norm(exp):
            problems.append(f"{task}: policy says {norm(want)}, task expects {norm(exp)}")
        problems += [f"{task}: {k} is false" for k, v in checks.items() if v is False]
    return rows, problems


def oracle_of(acts):
    """The outcome a list of actions amounts to, in expected.json form."""
    out = {}
    for a in acts:
        if a["action"] == "refund": out.setdefault("refunds", {})[a["order_id"]] = sorted(a["item_ids"])
        if a["action"] == "cancel": out.setdefault("cancels", []).append(a["order_id"])
        if a["action"] == "address": out.setdefault("addresses", {})[a["order_id"]] = a["address"]
    return out


if __name__ == "__main__":
    rows, problems = check()
    for r in rows:
        marks = " ".join(f"{k}={'-' if v is None else ('ok' if v else 'NO')}" for k, v in r["checks"].items())
        print(f"{r['task']:30} {r['basis']:14} {'agrees' if r['agrees'] else 'DIFFERS'}  {marks}")
    print("problems:", problems or "none")
    raise SystemExit(1 if problems else 0)
