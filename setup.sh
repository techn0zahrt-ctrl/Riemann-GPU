#!/usr/bin/env bash
# Installs all dependencies on all 4 nodes and mounts the NAS.
# Run once as: bash setup.sh
set -e

NODES=(192.168.100.9 192.168.100.10 192.168.100.11 192.168.100.12)
NAS_HOST=192.168.100.200
NAS_SHARE=math
NAS_USER=math
NAS_PASS=math
MOUNT=/mnt/math
WORK_DIR=/home/techno/math

# ── Per-node installation ─────────────────────────────────────────────────────

install_node() {
    local host=$1
    echo "=== Setting up $host ==="
    ssh "$host" bash -s <<'REMOTE'
set -e

# WSL: make nvidia-smi / CUDA libs discoverable
if [ -d /usr/lib/wsl/lib ] && ! echo "$PATH" | grep -q wsl; then
    grep -q 'wsl/lib' ~/.bashrc 2>/dev/null || \
        echo 'export PATH="/usr/lib/wsl/lib:$PATH"' >> ~/.bashrc
    export PATH="/usr/lib/wsl/lib:$PATH"
fi

# System packages
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3-pip python3-dev build-essential \
    cifs-utils smbclient \
    git curl wget 2>&1 | grep -E '(Install|Unpacking|already)'

# Python packages - try CUDA-enabled cupy first, fall back gracefully
pip3 install --quiet --upgrade pip
pip3 install --quiet numpy scipy mpmath tqdm

# Try cupy for CUDA 13 (package name changed in newer cupy releases)
pip3 install --quiet cupy-cuda12x 2>/dev/null \
    || pip3 install --quiet cupy 2>/dev/null \
    || echo "  [warn] cupy not installed - will use CPU path"

echo "  [ok] $HOSTNAME done"
REMOTE
}

# Run installs in parallel
for node in "${NODES[@]}"; do
    install_node "$node" &
done
wait
echo "All nodes installed."

# ── Mount NAS on all nodes ────────────────────────────────────────────────────

mount_nas() {
    local host=$1
    echo "=== Mounting NAS on $host ==="
    ssh "$host" bash -s <<REMOTE
set -e
sudo mkdir -p $MOUNT
# Write credentials file (keeps password off command line)
echo "username=$NAS_USER
password=$NAS_PASS" | sudo tee /etc/.math_creds > /dev/null
sudo chmod 600 /etc/.math_creds

# Unmount if stale
sudo umount $MOUNT 2>/dev/null || true

sudo mount -t cifs //$NAS_HOST/$NAS_SHARE $MOUNT \
    -o credentials=/etc/.math_creds,uid=\$(id -u),gid=\$(id -g),\
file_mode=0664,dir_mode=0775,vers=3.0

echo "  [ok] Mounted $NAS_HOST/$NAS_SHARE → $MOUNT"
REMOTE
}

for node in "${NODES[@]}"; do
    mount_nas "$node" &
done
wait

# ── Create result directories on NAS ─────────────────────────────────────────

echo "Creating NAS directory structure..."
mkdir -p $MOUNT/rh/{queue,active,done,logs}
cat > $MOUNT/rh/config.json <<JSON
{
  "T_start":   3000000000000.0,
  "T_end":     9999999999999.0,
  "dt":        0.05,
  "chunk_dt":  5000.0,
  "note": "Extend RH zero verification beyond ~3e12. Backlund method."
}
JSON
echo "NAS ready."

# ── Copy code to all worker nodes ─────────────────────────────────────────────

for node in 192.168.100.10 192.168.100.11 192.168.100.12; do
    echo "Copying code to $node..."
    ssh "$node" "mkdir -p $WORK_DIR"
    scp -q "$WORK_DIR"/*.py "$node:$WORK_DIR/"
done

echo ""
echo "Setup complete. Run:  bash $WORK_DIR/launch.sh"
