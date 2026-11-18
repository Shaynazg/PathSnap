import json
import os
from pathlib import Path
from threading import Lock

PATH = Path(os.getenv("PATHSNAP_LOG_FILE", Path(__file__).parent / "data" / "pathsnap_logs.jsonl"))
_lock = Lock()

def append(event: dict) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock, PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")

def recent(filters: dict) -> list[dict]:
    if not PATH.exists(): return []
    with _lock, PATH.open(encoding="utf-8") as stream:
        events = [json.loads(line) for line in stream if line.strip()]
    for key in ("severity", "source", "event_type", "trip_id"):
        if filters.get(key): events = [e for e in events if e.get(key) == filters[key]]
    return list(reversed(events[-filters.get("limit", 100):]))
