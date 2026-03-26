# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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
    JL_API_KEY=<token> python tools/gpu_launch_jarvislabs.py create
    JL_API_KEY=<token> python tools/gpu_launch_jarvislabs.py list
    JL_API_KEY=<token> python tools/gpu_launch_jarvislabs.py destroy <instance_id>
"""

from __future__ import annotations

import sys


def get_client():
    from jarvislabs.jarvisclient import Client

    return Client()


def cmd_balance():
    c = get_client()
    balance = c.account.balance()
    print(f"Balance: ${balance}")
    return balance


def cmd_list():
    c = get_client()
    instances = c.instances.list()
    if not instances:
        print("No running instances.")
        return
    for inst in instances:
        print(
            f"  ID={inst.id}  Name={inst.name}"
            f"  Status={inst.status}  GPU={inst.gpu_type}"
        )


def cmd_gpu_availability():
    c = get_client()
    gpus = c.account.gpu_availability()
    print("Available GPUs:")
    for gpu in gpus:
        print(f"  {gpu}")


def cmd_create():
    c = get_client()

    balance = c.account.balance()
    print(f"Balance: ${balance}")
    if float(str(balance).replace("$", "")) < 1:
        print("ERROR: Balance too low. Top up at https://cloud.jarvislabs.ai")
        return

    print("Creating L4 instance (24GB VRAM, ~$0.30/hr)...")
    print("If L4 unavailable, will try A5000Pro.")

    try:
        inst = c.instances.create(
            name="spo-benchmark",
            gpu_type="L4",
            num_gpus=1,
            storage=50,  # 50GB — JAX cache + pip + repo need headroom
            framework="pytorch",  # includes CUDA, we'll install JAX on top
        )
        print(f"Instance created: ID={inst.id}")
        print(f"Status: {inst.status}")
        print(f"SSH: ssh user@{inst.ssh_host} -p {inst.ssh_port}")
        print()
        print("NEXT STEPS:")
        print(f"  1. SSH in: ssh user@{inst.ssh_host} -p {inst.ssh_port}")
        print("  2. Run: bash tools/gpu_instance_setup.sh")
        print("  3. Run: python tools/gpu_benchmark.py")
        print("  4. Download results BEFORE destroying")
    except Exception as e:
        print(f"L4 creation failed: {e}")
        print("Trying A5000Pro...")
        try:
            inst = c.instances.create(
                name="spo-benchmark",
                gpu_type="A5000Pro",
                num_gpus=1,
                storage=50,
                framework="pytorch",
            )
            print(f"Instance created: ID={inst.id}")
            print(f"SSH: ssh user@{inst.ssh_host} -p {inst.ssh_port}")
        except Exception as e2:
            print(f"A5000Pro also failed: {e2}")
            print("Check https://cloud.jarvislabs.ai for available GPUs.")


def cmd_destroy(instance_id: str):
    print("=" * 50)
    print("WARNING: DESTROYING INSTANCE")
    print("=" * 50)
    print()
    print("Have you downloaded the benchmark results?")
    print("  scp user@<IP>:~/scpn-phase-orchestrator/benchmarks/results/*.json .")
    print()
    answer = input("Type 'yes' to confirm destroy: ")
    if answer.strip().lower() != "yes":
        print("Aborted.")
        return

    c = get_client()
    c.instances.destroy(instance_id)
    print(f"Instance {instance_id} destroyed.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/gpu_launch_jarvislabs.py <command>")
        print("Commands: balance, list, gpus, create, destroy <id>")
        return

    cmd = sys.argv[1]
    if cmd == "balance":
        cmd_balance()
    elif cmd == "list":
        cmd_list()
    elif cmd == "gpus":
        cmd_gpu_availability()
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
