# hitl.py
import json
import os
from datetime import datetime

ESCALATION_LOG = "escalation_log.json"

def _load_log() -> list:
    """Internal helper — loads the escalation log from disk"""
    if not os.path.exists(ESCALATION_LOG):
        return []
    with open(ESCALATION_LOG, "r") as f:
        return json.load(f)

def _save_log(log: list):
    """Internal helper — saves the escalation log to disk"""
    with open(ESCALATION_LOG, "w") as f:
        json.dump(log, f, indent=2)

def log_escalation(query: str, reason: str) -> dict:
    """
    Log an escalated query for human agent review.
    Skips if the same query is already pending — no duplicates.
    """
    log = _load_log()

    # Duplicate check — don't log same query twice while still pending
    already_pending = any(
        e["query"] == query and e["status"] == "pending"
        for e in log
    )
    if already_pending:
        print(f"⚠️ Query already pending escalation, skipping duplicate log")
        return {}

    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "reason": reason,
        "status": "pending",
        "resolved_at": None,
        "human_reply": None
    }

    log.append(entry)
    _save_log(log)
    print(f"📋 Escalation logged — reason: {reason}")
    return entry

def get_pending_escalations() -> list:
    """Fetch all queries waiting for human agent review"""
    log = _load_log()
    return [e for e in log if e["status"] == "pending"]

def get_resolved_escalations() -> list:
    """Fetch all queries that have been resolved by a human agent"""
    log = _load_log()
    return [e for e in log if e["status"] == "resolved"]

def resolve_escalation(timestamp: str, human_reply: str) -> bool:
    """
    Mark a query as resolved with the human agent's reply.
    Returns True if found and resolved, False if timestamp not found.
    """
    log = _load_log()
    resolved = False

    for entry in log:
        if entry["timestamp"] == timestamp:
            entry["status"] = "resolved"
            entry["human_reply"] = human_reply
            entry["resolved_at"] = datetime.now().isoformat()
            resolved = True
            break

    if resolved:
        _save_log(log)
        print(f"✅ Escalation resolved at {datetime.now().isoformat()}")
    else:
        print(f"❌ No escalation found with timestamp: {timestamp}")

    return resolved

def get_escalation_stats() -> dict:
    """
    Summary of escalation log — useful for sidebar display in app.py
    Returns counts of pending, resolved and total
    """
    log = _load_log()
    pending = sum(1 for e in log if e["status"] == "pending")
    resolved = sum(1 for e in log if e["status"] == "resolved")
    return {
        "total": len(log),
        "pending": pending,
        "resolved": resolved
    }