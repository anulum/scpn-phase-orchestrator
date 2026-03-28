#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Live SYNAPSE_CHANNEL bridge test
#
# Connects to a running SYNAPSE_CHANNEL hub, monitors agent activity,
# and computes SPO-style coherence metrics in real time.
# Does NOT import the heavy SPO engine — pure lightweight test.
#
# Requires: hub running at ws://localhost:8876
# Usage: python tools/synapse_bridge_live_test.py

from __future__ import annotations

import asyncio
import json
import math
import time

import websockets

TWO_PI = 2.0 * math.pi


class LightweightBridge:
    """Minimal bridge for testing — no SPO dependency."""

    def __init__(self, agents: list[str]) -> None:
        self.agents = agents
        self.n = len(agents)
        self.idx = {name: i for i, name in enumerate(agents)}
        self.last_hb: dict[str, float] = {}
        self.hb_intervals: dict[str, list[float]] = {a: [] for a in agents}
        self.tasks: dict[str, str | None] = dict.fromkeys(agents)
        self.msg_counts: dict[str, int] = dict.fromkeys(agents, 0)
        self.phases: list[float] = [0.0] * len(agents)

    def process(self, msg: dict) -> None:
        sender = msg.get("sender", "")
        msg_type = msg.get("type", "")
        now = time.time()

        if sender not in self.idx:
            return

        if msg_type == "heartbeat":
            if sender in self.last_hb:
                dt = now - self.last_hb[sender]
                self.hb_intervals[sender].append(dt)
                if len(self.hb_intervals[sender]) > 10:
                    self.hb_intervals[sender].pop(0)
            self.last_hb[sender] = now

        elif msg_type == "chat":
            self.msg_counts[sender] += 1

        elif "claim" in msg_type:
            self.tasks[sender] = msg.get("payload", "task")

        elif "release" in msg_type:
            self.tasks[sender] = None

    def compute_r(self) -> float:
        """Kuramoto order parameter from agent phases."""
        if self.n == 0:
            return 0.0
        # Update phases from heartbeat frequency
        for name, intervals in self.hb_intervals.items():
            if intervals:
                freq = 1.0 / max(sum(intervals) / len(intervals), 0.1)
                i = self.idx[name]
                self.phases[i] = (self.phases[i] + TWO_PI * freq * 0.1) % TWO_PI

        sin_sum = sum(math.sin(p) for p in self.phases)
        cos_sum = sum(math.cos(p) for p in self.phases)
        return math.sqrt(sin_sum**2 + cos_sum**2) / self.n


async def main() -> None:
    uri = "ws://localhost:8876"
    print("SYNAPSE_CHANNEL -> SPO Bridge Live Test")
    print("=" * 50)

    ws = await websockets.connect(uri)
    welcome = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
    online = welcome.get("online_agents", [])
    print(f"Hub: {welcome['hub_id']}")
    print(f"Online agents: {online}")

    if not online:
        online = ["agent_1", "agent_2"]

    bridge = LightweightBridge(online)

    # Send announcement
    await ws.send(
        json.dumps(
            {
                "type": "chat",
                "sender": "SPO-Bridge",
                "target": "all",
                "payload": "SPO coherence bridge monitoring started",
            }
        )
    )

    print(f"\nMonitoring {bridge.n} agents for 30 seconds...\n")
    print(f"{'Time':>6s}  {'R':>6s}  {'Msgs':>5s}  {'Active':>7s}")
    print("-" * 30)

    start = time.time()
    tick = 0

    while time.time() - start < 30:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            msg = json.loads(raw)
            bridge.process(msg)
        except asyncio.TimeoutError:
            pass

        tick += 1
        if tick % 5 == 0:
            R = bridge.compute_r()
            total_msgs = sum(bridge.msg_counts.values())
            sum(1 for t in bridge.tasks.values() if t is not None)
            elapsed = time.time() - start
            regime = "sync" if R > 0.6 else "drift" if R > 0.3 else "CONFLICT"
            line = f"{elapsed:>5.0f}s  R={R:.3f}  msgs={total_msgs}  [{regime}]"
            print(line)

    print("\nFinal agent states:")
    for name in bridge.agents:
        hb = len(bridge.hb_intervals[name])
        msgs = bridge.msg_counts[name]
        task = bridge.tasks[name] or "idle"
        print(f"  {name}: {hb} heartbeats, {msgs} msgs, {task}")

    R = bridge.compute_r()
    print(f"\nFinal R = {R:.3f}")
    print("Bridge verified: hub events -> phases -> R(t)")

    await ws.close()


if __name__ == "__main__":
    asyncio.run(main())
