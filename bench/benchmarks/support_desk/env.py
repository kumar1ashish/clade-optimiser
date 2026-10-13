"""The store the support agent works on. All state lives in the episode workspace: db.json (customers, orders,
"today"), policy.md (the refund and account policy), and actions.jsonl (every side effect, appended by the tools).
Hidden tests read db.json and actions.jsonl after the episode, so scoring is deterministic and needs no judge model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

TOOLS = [
    {"type": "function", "function": {"name": "lookup_customer", "description": "Find a customer account by email. Returns the account and its order ids.",
                                      "parameters": {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]}}},
    {"type": "function", "function": {"name": "get_order", "description": "Get one order: status, delivery date and items (price, opened, damaged, final_sale).",
                                      "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "search_policy", "description": "Search the store policy. Returns the matching sections.",
                                      "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "refund_items", "description": "Refund specific items of a delivered order to the original payment method.",
                                      "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "item_ids": {"type": "array", "items": {"type": "string"}},
                                                                                      "reason": {"type": "string"}}, "required": ["order_id", "item_ids", "reason"]}}},
    {"type": "function", "function": {"name": "issue_store_credit", "description": "Give a customer store credit.",
                                      "parameters": {"type": "object", "properties": {"customer_id": {"type": "string"}, "amount": {"type": "number"},
                                                                                      "reason": {"type": "string"}}, "required": ["customer_id", "amount", "reason"]}}},
    {"type": "function", "function": {"name": "cancel_order", "description": "Cancel an order that has not shipped yet.",
                                      "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "update_address", "description": "Change the shipping address of an order.",
                                      "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "address": {"type": "string"}}, "required": ["order_id", "address"]}}},
    {"type": "function", "function": {"name": "escalate", "description": "Hand the case to a human specialist.",
                                      "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}},
    {"type": "function", "function": {"name": "reply_to_customer", "description": "Send the reply to the customer. Ends the conversation.",
                                      "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}}},
]


class Store:
    def __init__(self, cwd: Path):
        self.cwd = Path(cwd)
        self.db = json.loads((self.cwd / "db.json").read_text())

    def _save(self) -> None:
        (self.cwd / "db.json").write_text(json.dumps(self.db, indent=1))

    def _log(self, action: str, **data) -> None:
        with (self.cwd / "actions.jsonl").open("a") as f:
            f.write(json.dumps({"action": action, **data}) + "\n")

    def _order(self, order_id: str) -> dict:
        o = self.db["orders"].get(str(order_id).strip())
        if o is None:
            raise ValueError(f"no order {order_id}")
        return o

    # ---- tools ----
    def lookup_customer(self, email: str) -> dict:
        email = str(email).strip().lower()
        for cid, c in self.db["customers"].items():
            if c["email"].lower() == email:
                return {"customer_id": cid, **c, "order_ids": [oid for oid, o in self.db["orders"].items() if o["customer_id"] == cid]}
        return {"error": f"no account with email {email}"}

    def get_order(self, order_id: str) -> dict:
        return {"order_id": order_id, **self._order(order_id), "today": self.db["today"]}

    def search_policy(self, query: str) -> str:
        sections = re.split(r"\n(?=## )", (self.cwd / "policy.md").read_text())
        words = [w for w in re.findall(r"[a-z]+", str(query).lower()) if len(w) > 2]
        scored = sorted(sections, key=lambda s: -sum(s.lower().count(w) for w in words))
        hits = [s for s in scored if any(w in s.lower() for w in words)][:3]
        return "\n\n".join(hits) if hits else "No matching policy section. Sections: " + ", ".join(re.findall(r"## (.+)", "\n".join(sections)))

    def refund_items(self, order_id: str, item_ids: list, reason: str = "") -> dict:
        o = self._order(order_id)
        if o["status"] != "delivered":
            raise ValueError(f"order {order_id} is {o['status']}; only delivered orders can be refunded")
        ids = [str(i) for i in (item_ids if isinstance(item_ids, list) else [item_ids])]
        items = {i["id"]: i for i in o["items"]}
        bad = [i for i in ids if i not in items]
        if bad:
            raise ValueError(f"order {order_id} has no items {bad}")
        already = [i for i in ids if items[i].get("refunded")]
        if already:
            raise ValueError(f"items {already} were already refunded")
        amount = round(sum(items[i]["price"] for i in ids), 2)
        for i in ids:
            items[i]["refunded"] = True
        self._save(); self._log("refund", order_id=order_id, item_ids=sorted(ids), amount=amount, reason=reason)
        return {"refunded": sorted(ids), "amount": amount}

    def issue_store_credit(self, customer_id: str, amount: float, reason: str = "") -> dict:
        c = self.db["customers"].get(str(customer_id))
        if c is None:
            raise ValueError(f"no customer {customer_id}")
        c["store_credit"] = round(c.get("store_credit", 0) + float(amount), 2)
        self._save(); self._log("store_credit", customer_id=customer_id, amount=round(float(amount), 2), reason=reason)
        return {"customer_id": customer_id, "store_credit": c["store_credit"]}

    def cancel_order(self, order_id: str) -> dict:
        o = self._order(order_id)
        if o["status"] != "processing":
            raise ValueError(f"order {order_id} is {o['status']}; only processing orders can be cancelled")
        o["status"] = "cancelled"
        self._save(); self._log("cancel", order_id=order_id)
        return {"order_id": order_id, "status": "cancelled"}

    def update_address(self, order_id: str, address: str) -> dict:
        o = self._order(order_id)
        o["shipping_address"] = str(address)
        self._save(); self._log("address", order_id=order_id, address=str(address))
        return {"order_id": order_id, "shipping_address": o["shipping_address"]}

    def escalate(self, reason: str) -> dict:
        self._log("escalate", reason=str(reason))
        return {"escalated": True}

    def reply_to_customer(self, message: str) -> dict:
        self._log("reply", message=str(message))
        return {"sent": True}

    def call(self, name: str, args: dict) -> str:
        fn = getattr(self, name, None)
        if name.startswith("_") or name == "call" or fn is None:
            raise ValueError(f"unknown tool {name}")
        out = fn(**args)
        return out if isinstance(out, str) else json.dumps(out)
