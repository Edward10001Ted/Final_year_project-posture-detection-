import builtins
import time
import threading
import copy
import json
import os

from sena_status import Sena as SharedSena, get_sena_status, publish_status

# Shared container requested by you
Sena = SharedSena

# Preserve original print
_original_print = builtins.print

# Buffer of recent printed lines (timestamp_ms, text)
_print_buffer = []
_buffer_lock = threading.Lock()

# Hook print to capture terminal text
def _hooked_print(*args, **kwargs):
    try:
        _original_print(*args, **kwargs)
    except Exception:
        pass
    try:
        text = ' '.join(str(a) for a in args)
    except Exception:
        text = repr(args)
    ts_ms = int(time.time() * 1000)
    with _buffer_lock:
        _print_buffer.append((ts_ms, text))

# Install hook before importing ai_engine so we capture its startup prints
builtins.print = _hooked_print

# Import the engine (this will start its background AI thread)
import ai_engine

# Snapshot loop: every N seconds, copy LIVE_OUTPUT_STREAM and buffered prints into Sena
def _snapshot_loop(interval: float = 60.0):
    while True:
        time.sleep(interval)
        ts_ms = str(int(time.time() * 1000))
        with _buffer_lock:
            prints = list(_print_buffer)
            _print_buffer.clear()
        try:
            live = copy.deepcopy(ai_engine.LIVE_OUTPUT_STREAM)
        except Exception:
            live = {}
        status_payload = publish_status(live)
        Sena[ts_ms] = {
            "prints": prints,
            "live": live,
            "status": status_payload,
        }
        Sena["latest_status"] = status_payload
        Sena["latest_status_timestamp"] = status_payload.get("timestamp")
        try:
            _dump_sena_atomic()
        except Exception:
            pass
        try:
            _original_print(f"[sena_logger] Snapshot {ts_ms}: workers={len(live.get('workers', {}))}")
        except Exception:
            pass

_snapshot_thread = threading.Thread(target=_snapshot_loop, daemon=True)
_snapshot_thread.start()

_original_print("sena_logger started — global variable `Sena` will update every 60s and expose `latest_status` for backend import.")

if __name__ == "__main__":
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _original_print("sena_logger exiting.")


# Persist Sena to disk atomically
SAVE_PATH = os.path.join(os.path.dirname(__file__), "sena_dump.json")

def _dump_sena_atomic():
    try:
        sena_copy = copy.deepcopy(Sena)
        tmp_path = SAVE_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(sena_copy, f, default=str, ensure_ascii=False, indent=2)
        os.replace(tmp_path, SAVE_PATH)
    except Exception as e:
        _original_print(f"[sena_logger] Failed to dump Sena: {e}")

# Optionally create an initial dump so the file exists
try:
    _dump_sena_atomic()
except Exception:
    pass
