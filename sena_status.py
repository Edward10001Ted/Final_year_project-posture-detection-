import json
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional


_STATUS_LOCK = threading.Lock()
Sena: Dict[str, Any] = {
    "working_status": "UNKNOWN",
    "timestamp": "",
    "timestamp_epoch": 0,
    "active_workers_count": 0,
    "working_workers_count": 0,
    "workers": {},
}
_CURRENT_STATUS: Dict[str, Any] = Sena


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_status_payload(live_stream: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    live = live_stream or {}
    workers = live.get("workers") or {}
    worker_items = list(workers.values()) if isinstance(workers, dict) else []

    working_workers = [
        worker for worker in worker_items
        if isinstance(worker, dict) and str(worker.get("status", "")).upper() == "WORKING"
    ]

    payload = {
        "working_status": "WORKING" if working_workers else "NOT-WORKING",
        "timestamp": _now_iso(),
        "timestamp_epoch": int(time.time()),
        "active_workers_count": len(worker_items),
        "working_workers_count": len(working_workers),
        "workers": workers,
    }
    return payload


def publish_status(live_stream: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = build_status_payload(live_stream)
    with _STATUS_LOCK:
        _CURRENT_STATUS.update(payload)
        Sena.update(payload)
    return payload


def get_sena_status() -> Dict[str, Any]:
    with _STATUS_LOCK:
        return dict(Sena)


def get_sena_status_json() -> str:
    return json.dumps(get_sena_status(), ensure_ascii=False, default=str)


def start_background_publisher(interval_seconds: float = 60.0, getter=None):
    """Periodically publish the latest status every minute by default."""

    def _runner() -> None:
        while True:
            try:
                source = getter() if getter is not None else None
                publish_status(source)
            except Exception:
                publish_status({})
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    return thread
