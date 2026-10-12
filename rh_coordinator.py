#!/usr/bin/env python3
"""
RH Coordinator: persistent-worker architecture.
One long-lived SSH process per slot; chunks stream through stdin/stdout
so GPU memory stays allocated and there is no inter-chunk idle time.

Run on master:  python3 rh_coordinator.py
"""

import collections
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rh_core import backlund_count
import nas_fs as nas

NODES = [
    "192.168.100.9",
    "192.168.100.10",
    "192.168.100.11",
    "192.168.100.12",
]
WORKERS_PER_NODE = 3
WORK_DIR         = "/home/techno/math"
STALE_TIMEOUT    = 7200
MAX_INFLIGHT     = 20   # chunks streaming in-flight per slot before back-pressure


# ── Shared state ──────────────────────────────────────────────────────────────

_pending:  collections.deque = collections.deque()
_results:  queue.Queue       = queue.Queue()   # (chunk_id, result_dict | None)
_lock:     threading.Lock    = threading.Lock()
_inflight: dict              = {}              # chunk_id → {node, started, chunk}

# In-memory running totals — updated by collect_results() to avoid NAS scans in write_summary()
_zeros_total: int = 0
_chunks_ok:   int = 0
_chunks_err:  int = 0
_chunks_done: int = 0  # only chunks completed THIS coordinator run


# ── Work queue ────────────────────────────────────────────────────────────────

def build_queue(config) -> int:
    done_ids   = {p.stem for p in nas.rh_path("done").iterdir()   if p.suffix == ".json"}
    active_ids = {p.stem for p in nas.rh_path("active").iterdir() if p.suffix == ".json"}
    _pending.clear()
    t = config["T_start"]
    idx = 0
    while t < config["T_end"]:
        t_end = min(t + config["chunk_dt"], config["T_end"])
        cid   = f"chunk_{idx:07d}"
        if cid not in done_ids and cid not in active_ids:
            _pending.append({"chunk_id": cid, "t_start": t,
                             "t_end": t_end, "dt": config["dt"]})
        t += config["chunk_dt"]
        idx += 1
    print(f"Queue: {len(_pending)} pending  |  {len(done_ids)} done  {len(active_ids)} active")
    return len(done_ids)


# ── Persistent worker slot ────────────────────────────────────────────────────

class WorkerSlot:
    """Persistent SSH worker with separate send/receive threads.

    The send thread feeds chunks to the worker's stdin as fast as it can
    accept them (up to MAX_INFLIGHT in-flight at once).  The receive thread
    reads results from stdout continuously — it never blocks the send thread.
    Together they ensure the remote GPU pipeline is always fed and the
    coordinator never waits idle for a round-trip to complete.
    """

    def __init__(self, node: str, slot: int):
        self.node       = node
        self.slot       = slot
        self.proc       = None
        self.log_f      = None
        self._inflight  = 0          # chunks sent but result not yet received
        self._lock      = threading.Lock()
        self._send_q: queue.Queue = queue.Queue()
        self._send_t = threading.Thread(target=self._send_loop, daemon=True,
                                        name=f"{node}:{slot}:send")
        self._recv_t = threading.Thread(target=self._recv_loop, daemon=True,
                                        name=f"{node}:{slot}:recv")
        self._send_t.start()
        self._recv_t.start()

    def _spawn(self):
        binary = f"{WORK_DIR}/rh_worker_cuda"
        cmd    = (f"export LD_LIBRARY_PATH={WORK_DIR}:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH; "
                  f"export PATH=/usr/lib/wsl/lib:$PATH; "
                  f"if [ -x {binary} ]; then {binary}; "
                  f"else python3 {WORK_DIR}/rh_worker.py; fi")
        return subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             "-o", "ServerAliveInterval=30", self.node, cmd],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log_f or subprocess.DEVNULL,
        )

    def _send_loop(self):
        """Sends individual chunks to the worker's stdin.  Spawns/respawns as needed."""
        while True:
            chunk = self._send_q.get()
            if chunk is None:
                return
            if self.proc is None or self.proc.poll() is not None:
                self.proc = self._spawn()
            nas.write_json(nas.rh_path("active", f"{chunk['chunk_id']}.json"), chunk)
            try:
                self.proc.stdin.write((json.dumps(chunk) + "\n").encode())
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                _pending.appendleft(chunk)
                nas.rh_path("active", f"{chunk['chunk_id']}.json").unlink(missing_ok=True)
                with _lock:
                    _inflight.pop(chunk["chunk_id"], None)
                with self._lock:
                    self._inflight = max(0, self._inflight - 1)
                self.proc = None

    def _recv_loop(self):
        """Reads results from stdout continuously — never blocks the send thread."""
        while True:
            p = self.proc
            if p is None:
                time.sleep(0.02)
                continue
            try:
                out = p.stdout.readline()
            except OSError:
                self.proc = None
                continue
            if not out:
                # EOF: worker died — reset inflight; stale reclaim will requeue
                with self._lock:
                    lost = self._inflight
                    self._inflight = 0
                if lost:
                    print(f"  [slot {self.node}:{self.slot}] worker died ({lost} in-flight; reclaim via timeout)")
                self.proc = None
                time.sleep(1)
                continue
            try:
                result = json.loads(out.decode().strip())
                cid = result.get("chunk_id", "")
                _results.put((cid, result))
                with self._lock:
                    self._inflight = max(0, self._inflight - 1)
            except Exception as e:
                print(f"  [slot {self.node}:{self.slot}] parse error: {e}")

    def submit(self, chunk: dict) -> bool:
        """Non-blocking: send one chunk if under the inflight cap."""
        with self._lock:
            if self._inflight >= MAX_INFLIGHT:
                return False
            self._inflight += 1
        self._send_q.put_nowait(chunk)
        return True

    def is_free(self) -> bool:
        with self._lock:
            return self._inflight < MAX_INFLIGHT


# ── Result collector ──────────────────────────────────────────────────────────

def collect_results():
    global _zeros_total, _chunks_ok, _chunks_err, _chunks_done
    while not _results.empty():
        cid, result = _results.get_nowait()
        with _lock:
            _inflight.pop(cid, None)
        # The binary writes done/ and removes active/ directly before writing stdout.
        # Skip redundant NAS I/O here — each call was ~100ms and stalled chunk dispatch.
        ok = result.get("backlund_ok", False)
        zf = result.get("zeros_found", 0)
        _zeros_total += zf
        _chunks_done += 1
        if ok:
            _chunks_ok += 1
        else:
            _chunks_err += 1
        status = "✓" if ok else "✗"
        print(f"  {status}  {cid}  zeros={zf:,d}  "
              f"host={result.get('host','?')}  wall={result['wall_seconds']:.1f}s")


_last_reclaim: float = 0.0

def reclaim_stale():
    global _last_reclaim
    now = time.time()
    if now - _last_reclaim < 300:  # only every 5 minutes — NAS iterdir is slow
        return
    _last_reclaim = now
    for path in list(nas.rh_path("active").iterdir()):
        if path.suffix != ".json":
            continue
        try:
            age = now - path.stat().st_mtime
        except FileNotFoundError:
            continue
        if age > STALE_TIMEOUT:
            try:
                chunk = nas.read_json(path)
                path.unlink()
                _pending.appendleft(chunk)
                print(f"  [reclaim] {path.stem} stale, re-queued")
            except Exception:
                pass


# ── Summary ───────────────────────────────────────────────────────────────────

def write_summary(config, slots, n_done_start: int):
    # Use in-memory counters — reading all done/ files from NAS was blocking for 10s+
    n_done   = n_done_start + _chunks_done
    n_queue  = len(_pending)
    n_active = sum(1 for s in slots if not s.is_free())
    t_cover  = config["T_start"] + n_done * config["chunk_dt"]

    nas.write_json(nas.rh_path("summary.json"), {
        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "zeros_verified": _zeros_total,
        "coverage_t":     t_cover,
        "chunks_done":    n_done,
        "chunks_ok":      _chunks_ok,
        "chunks_error":   _chunks_err,
        "chunks_queued":  n_queue,
        "chunks_active":  n_active,
    })

    print(f"\n{'─'*60}")
    print(f"  Zeros verified  : {_zeros_total:,d}")
    print(f"  Coverage        : t ≈ {t_cover:.6e}")
    print(f"  Chunks          : {n_done} done  {n_queue} queued  {n_active} active")
    print(f"{'─'*60}\n")
    return n_done


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"RH Coordinator on {socket.gethostname()}  (persistent-worker mode)")

    if not nas.nas_status():
        sys.exit("NAS not mounted at /mnt/math")

    nas.ensure_rh_dirs()
    nas.ensure_default_config()
    config = nas.read_config()

    print(f"  T ∈ [{config['T_start']:.4e}, {config['T_end']:.4e}]  chunk_dt={config['chunk_dt']}")
    expected = backlund_count(config["T_end"]) - backlund_count(config["T_start"])
    print(f"  Zeros to verify : ≈{expected:,d}")

    n_done_start = build_queue(config)

    # Spawn persistent worker slots
    slots = [WorkerSlot(node, s)
             for node in NODES
             for s in range(WORKERS_PER_NODE)]
    print(f"  Slots: {len(slots)} ({len(NODES)} nodes × {WORKERS_PER_NODE})")

    last_summary = 0.0
    try:
        while True:
            collect_results()
            reclaim_stale()

            # Feed free slots — stream individual chunks up to MAX_INFLIGHT per slot
            for slot in slots:
                while slot.is_free() and _pending:
                    chunk = _pending.popleft()
                    with _lock:
                        _inflight[chunk["chunk_id"]] = {
                            "node": slot.node, "started": time.time(), "chunk": chunk}
                    if not slot.submit(chunk):
                        _pending.appendleft(chunk)
                        with _lock:
                            _inflight.pop(chunk["chunk_id"], None)
                        break

            if time.time() - last_summary >= 30:
                write_summary(config, slots, n_done_start)
                last_summary = time.time()

            if not _pending and all(s.is_free() for s in slots) and _results.empty():
                collect_results()
                if not _inflight:
                    print("All chunks complete!")
                    break

            time.sleep(0.05)  # reduced from 0.2s — less idle time per slot

    except KeyboardInterrupt:
        print("\nCoordinator stopped.")
        for s in slots:
            s._send_q.put_nowait(None)


if __name__ == "__main__":
    main()
