"""
baselines.py
------------
Implements three baseline offloading strategies for comparison
against the proposed Decision Engine.

Baselines:
  1. AlwaysLocal    — never offloads, always executes on device
  2. AlwaysOffload  — always sends task to edge server
  3. RandomBaseline — randomly decides with 50/50 probability

Project: BTP2CSE054 - Task Offloading Decision System in MEC
"""

import random


class AlwaysLocal:
    """Baseline: execute every task locally on the mobile device."""

    name = "Always-Local"

    def decide(self, task, device, channel, edge_server):
        latency = device.compute_local_latency(task)
        energy  = device.compute_local_energy(task)
        return "local", latency, energy, latency + energy


class AlwaysOffload:
    """Baseline: offload every task to the edge server."""

    name = "Always-Offload"

    def decide(self, task, device, channel, edge_server):
        t_off   = channel.transmission_latency(task)
        e_off   = channel.transmission_energy(task)
        t_edge  = edge_server.compute_edge_latency(task)
        latency = t_off + t_edge
        energy  = e_off
        return "offload", latency, energy, latency + energy


class RandomBaseline:
    """Baseline: randomly choose local or offload with equal probability."""

    name = "Random"

    def decide(self, task, device, channel, edge_server):
        if random.random() < 0.5:
            latency = device.compute_local_latency(task)
            energy  = device.compute_local_energy(task)
            return "local", latency, energy, latency + energy
        else:
            t_off   = channel.transmission_latency(task)
            e_off   = channel.transmission_energy(task)
            t_edge  = edge_server.compute_edge_latency(task)
            latency = t_off + t_edge
            energy  = e_off
            return "offload", latency, energy, latency + energy


class GreedyLatency:
    """
    Baseline: always choose the option with lower latency,
    ignoring energy or cost. (Greedy approach from literature.)
    """

    name = "Greedy-Latency"

    def decide(self, task, device, channel, edge_server):
        local_latency   = device.compute_local_latency(task)
        offload_latency = (
            channel.transmission_latency(task) +
            edge_server.compute_edge_latency(task)
        )
        if local_latency <= offload_latency:
            energy = device.compute_local_energy(task)
            return "local", local_latency, energy, local_latency + energy
        else:
            energy = channel.transmission_energy(task)
            return "offload", offload_latency, energy, offload_latency + energy
