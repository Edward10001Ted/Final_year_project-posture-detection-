"""prediction_store: simple shared-memory store for realtime predictions.

Usage (writer - `face_ver.py`):
    import prediction_store as store
    store.write(label, score)

Usage (reader - other backend):
    import prediction_store as store
    pred = store.read()  # returns dict {'label':..., 'score':...} or None

This uses multiprocessing.shared_memory and stores a JSON string in a fixed-size buffer.
"""
import json
from multiprocessing import shared_memory

# Shared memory config
_NAME = 'face_pred_shm'
_SIZE = 4096
_shm = None


def _ensure():
    global _shm
    if _shm is None:
        try:
            _shm = shared_memory.SharedMemory(name=_NAME, create=True, size=_SIZE)
            # zero-init
            _shm.buf[:] = b'\x00' * _SIZE
        except FileExistsError:
            _shm = shared_memory.SharedMemory(name=_NAME, create=False)


def write(label, score):
    """Write a prediction into shared memory (overwrites previous)."""
    _ensure()
    payload = json.dumps({'label': label, 'score': float(score)})
    data = payload.encode('utf-8')
    if len(data) >= _SIZE:
        raise ValueError('Prediction JSON too large for shared memory')
    # write data and null-terminate
    _shm.buf[:len(data)] = data
    _shm.buf[len(data)] = 0


def read():
    """Read the latest prediction from shared memory. Returns dict or None."""
    _ensure()
    buf = _shm.buf.tobytes()
    if buf[0] == 0:
        return None
    # find null terminator
    try:
        end = buf.index(0)
    except ValueError:
        end = _SIZE
    raw = buf[:end].decode('utf-8')
    try:
        return json.loads(raw)
    except Exception:
        return None


def close():
    """Close the shared memory handle (writer or reader can call this)."""
    global _shm
    if _shm is not None:
        _shm.close()
        _shm = None


def unlink():
    """Unlink (destroy) the shared memory block. Only call when you want to remove it."""
    global _shm
    if _shm is not None:
        try:
            _shm.unlink()
        except FileNotFoundError:
            pass
        _shm = None
