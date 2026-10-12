#!/usr/bin/env bash
# Launch the RH Coordinator.  Workers are started automatically via SSH.
# No root/sudo required — all I/O is local + SSH.  NAS gets archived via Windows.
set -e

WORK_DIR=/home/techno/math
NODES=(192.168.100.10 192.168.100.11 192.168.100.12)

export PATH=/usr/lib/wsl/lib:$PATH

# ── Sync code to workers ───────────────────────────────────────────────────────
echo "Syncing code to worker nodes..."
for node in "${NODES[@]}"; do
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" "mkdir -p $WORK_DIR" &
done
wait
for node in "${NODES[@]}"; do
    scp -q "$WORK_DIR"/rh_core.py "$WORK_DIR"/rh_worker.py "$node:$WORK_DIR/" &
done
wait
echo "Done."

# ── Node health ────────────────────────────────────────────────────────────────
echo ""
echo "Node status:"
for node in 192.168.100.9 "${NODES[@]}"; do
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" "echo ok" 2>/dev/null | grep -q ok; then
        GPU=$( ssh "$node" \
            "export PATH=/usr/lib/wsl/lib:\$PATH; \
             /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total \
             --format=csv,noheader 2>/dev/null || echo 'no GPU detected'" )
        CUPY=$(ssh "$node" "python3 -c 'import cupy; print(\"cupy ok\")' 2>/dev/null || echo 'CPU only'")
        echo "  ✓  $node   $GPU   [$CUPY]"
    else
        echo "  ✗  $node   UNREACHABLE"
    fi
done

echo ""
echo "Starting coordinator...  (Ctrl-C to stop)"
echo ""
echo "To monitor a worker live:"
echo "  ssh <node> 'tail -f /tmp/rh_chunk_*.log'"
echo ""

python3 "$WORK_DIR/rh_coordinator.py"
