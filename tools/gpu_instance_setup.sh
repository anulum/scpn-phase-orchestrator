#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — GPU instance setup (JarvisLabs)
#
# Run this ONCE on a fresh JarvisLabs instance:
#   bash tools/gpu_instance_setup.sh
#
# It will:
#   1. Validate GPU is present
#   2. Clone the repo
#   3. Install SPO with JAX CUDA
#   4. Run a smoke test
#   5. Print next steps

set -euo pipefail

echo "=========================================="
echo "SPO GPU Instance Setup"
echo "=========================================="

# Step 1: GPU check
echo ""
echo "[1/5] Checking GPU..."
if ! nvidia-smi > /dev/null 2>&1; then
    echo "FATAL: No GPU found. nvidia-smi failed."
    echo "Check JarvisLabs instance settings."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "  GPU OK"

# Step 2: Disk check
echo ""
echo "[2/5] Checking disk..."
FREE_GB=$(df -BG / | tail -1 | awk '{print $4}' | tr -d 'G')
echo "  Free: ${FREE_GB}GB"
if [ "$FREE_GB" -lt 5 ]; then
    echo "WARNING: Low disk space. Need at least 5GB."
fi

# Step 3: Clone and install
echo ""
echo "[3/5] Cloning repo and installing..."
cd /home/user
if [ -d "scpn-phase-orchestrator" ]; then
    echo "  Repo exists, pulling latest..."
    cd scpn-phase-orchestrator
    git pull origin main
else
    git clone https://github.com/anulum/scpn-phase-orchestrator.git
    cd scpn-phase-orchestrator
fi

pip install --quiet --require-hashes --no-deps -r requirements/dev-lock.txt && pip install --quiet --no-deps -e . 2>&1 | tail -3
echo "  Install complete"

# Step 4: Smoke test
echo ""
echo "[4/5] Smoke test..."
python -c "
import jax
print(f'  JAX {jax.__version__}, devices: {jax.devices()}')
gpu = [d for d in jax.devices() if d.platform == \"gpu\"]
assert len(gpu) > 0, 'No GPU in JAX!'
print(f'  GPU: {gpu[0]}')

from scpn_phase_orchestrator.nn.functional import kuramoto_step
import jax.numpy as jnp
import jax.random as jr
key = jr.PRNGKey(0)
p = jr.uniform(key, (8,))
o = jnp.zeros(8)
K = jnp.ones((8,8)) * 0.1
a = jnp.zeros((8,8))
p2 = kuramoto_step(p, o, K, a, 0.01)
print(f'  kuramoto_step OK: {p2.shape}')
print('  SMOKE TEST PASSED')
"

# Step 5: Next steps
echo ""
echo "=========================================="
echo "SETUP COMPLETE"
echo "=========================================="
echo ""
echo "Run benchmarks:"
echo "  cd /home/user/scpn-phase-orchestrator"
echo "  python tools/gpu_benchmark.py"
echo ""
echo "Monitor:"
echo "  nvidia-smi -l 5  (GPU usage)"
echo "  watch -n 10 ls -la benchmarks/results/  (results)"
echo ""
echo "After benchmarks complete, download results BEFORE destroying:"
echo "  scp user@\$(hostname -I | awk '{print \$1}'):~/scpn-phase-orchestrator/benchmarks/results/*.json ."
