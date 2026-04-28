# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — JarvisLabs GPU instance launcher

"""Launch, monitor, and recover JarvisLabs GPU instances.

RECOVERY PLAN (read before running):

1. Script creates instance → prints SSH command + IP
2. You SSH in, run setup + benchmarks
3. Download results BEFORE any lifecycle operation
4. Only destroy after local verification

IF CLI FAILS MID-RUN:
- Instance keeps running (billed per hour)
- Log in to https://cloud.jarvislabs.ai to manage
- SSH still works — download data first
- API token: set JL_API_KEY env var

IF INSTANCE DISAPPEARS:
- Check JarvisLabs dashboard
- Results are lost ONLY if not downloaded
- Re-run is safe (benchmark script has resume/checkpoint)

COST ESTIMATE:
- L4 (24GB): ~$0.30/hr → $15 = 50 hours
- A5000Pro (24GB): ~$0.40/hr → $15 = 37 hours
- Our benchmarks take <30 minutes → ~$0.25 total

Usage:
    JL_API_KEY=<token> python tools/gpu_launch_jarvislabs.py balance
    JL_API_KEY=<token> python tools/gpu_launch_jarvislabs.py create
    JL_API_KEY=<token> python tools/gpu_launch_jarvislabs.py list
    JL_API_KEY=<token> python tools/gpu_launch_jarvislabs.py destroy <instance_id>
"""

from __future__ import annotations

import os
import sys


def _init_token() -> None:
    from jlclient import jarvisclient  # type: ignore[import-not-found]

    token = os.environ.get("JL_API_KEY", "")
    if not token:
        print("ERROR: Set JL_API_KEY environment variable")
        sys.exit(1)
    jarvisclient.token = token


def cmd_balance() -> object:
    _init_token()
    from jlclient.jarvisclient import User  # type: ignore[import-not-found]

    result = User.get_balance()
    balance = result.get("balance", result)
    print(f"Balance: ${balance}")
    return balance


def cmd_list() -> None:
    _init_token()
    from jlclient.jarvisclient import Instance

    instances = Instance.get_all()
    if not instances:
        print("No running instances.")
        return
    for inst in instances:
        print(f"  ID={inst.machine_id}  Name={inst.name}  Status={inst.status}")


def cmd_create() -> None:
    _init_token()
    from jlclient.jarvisclient import Instance, User

    result = User.get_balance()
    balance = result.get("balance", 0)
    print(f"Balance: ${balance}")
    if balance < 1:
        print("ERROR: Balance too low. Top up at https://cloud.jarvislabs.ai")
        return

    print("Creating L4 instance (24GB VRAM, ~$0.30/hr)...")

    try:
        inst = Instance.create(
            instance_type="on-demand",
            name="spo-benchmark",
            gpu_type="L4",
            num_gpus=1,
            storage=50,
            template="pytorch",
        )
        print(f"Instance created: ID={inst.machine_id}")
        print(f"Status: {inst.status}")
        print()
        print("NEXT STEPS:")
        print("  1. Wait ~2 min for instance to start")
        print("  2. Run 'list' command to get SSH details")
        print("  3. SSH in, clone repo, run benchmarks")
        print("  4. Download results BEFORE destroying")
    except Exception as e:
        print(f"L4 creation failed: {e}")
        print("Check https://cloud.jarvislabs.ai for available GPUs.")


def cmd_destroy(instance_id: str) -> None:
    _init_token()
    from jlclient.jarvisclient import Instance

    print("=" * 50)
    print("WARNING: DESTROYING INSTANCE")
    print("=" * 50)
    print()
    print("Have you downloaded the benchmark results?")
    print()
    answer = input("Type 'yes' to confirm destroy: ")
    if answer.strip().lower() != "yes":
        print("Aborted.")
        return

    Instance.delete(machine_id=int(instance_id))
    print(f"Instance {instance_id} destroyed.")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python tools/gpu_launch_jarvislabs.py <command>")
        print("Commands: balance, list, gpus, create, destroy <id>")
        return

    cmd = sys.argv[1]
    if cmd == "balance":
        cmd_balance()
    elif cmd == "list":
        cmd_list()
    elif cmd == "create":
        cmd_create()
    elif cmd == "destroy":
        if len(sys.argv) < 3:
            print("Usage: destroy <instance_id>")
            return
        cmd_destroy(sys.argv[2])
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
