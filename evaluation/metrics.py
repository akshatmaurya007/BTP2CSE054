"""
metrics.py
----------
Computes and reports performance metrics for the simulation.

Metrics tracked (per project synopsis):
  1. Average Task Latency        (target: < 50ms)
  2. Average Energy Consumption  (minimised vs baseline)
  3. Offloading Accuracy         (target: > 90%)
  4. Task Completion Rate        (target: > 95%)
  5. Resource Utilisation        (target: 40-70%)

Project: BTP2CSE054 - Task Offloading Decision System in MEC
"""

import numpy as np


class MetricsTracker:
    """
    Accumulates per-task results and computes aggregate metrics.
    """

    def __init__(self, strategy_name="Proposed"):
        self.strategy_name = strategy_name
        self.latencies  = []
        self.energies   = []
        self.costs      = []
        self.decisions  = []
        self.completed  = []     # True if task finished within deadline
        self.correct    = []     # True if decision matches ground-truth label

    def record(self, latency, energy, cost, decision, completed, correct=None):
        self.latencies.append(latency)
        self.energies.append(energy)
        self.costs.append(cost)
        self.decisions.append(decision)
        self.completed.append(completed)
        if correct is not None:
            self.correct.append(correct)

    def avg_latency_ms(self):
        return np.mean(self.latencies) * 1000 if self.latencies else 0

    def avg_energy(self):
        return np.mean(self.energies) if self.energies else 0

    def avg_cost(self):
        return np.mean(self.costs) if self.costs else 0

    def task_completion_rate(self):
        if not self.completed:
            return 0
        return sum(self.completed) / len(self.completed) * 100

    def offloading_accuracy(self):
        if not self.correct:
            return None
        return sum(self.correct) / len(self.correct) * 100

    def offload_ratio(self):
        if not self.decisions:
            return 0
        return sum(1 for d in self.decisions if d == "offload") / len(self.decisions) * 100

    def report(self):
        print(f"\n{'='*55}")
        print(f"  Strategy: {self.strategy_name}")
        print(f"  Tasks evaluated : {len(self.latencies)}")
        print(f"{'='*55}")
        print(f"  Avg Latency          : {self.avg_latency_ms():.2f} ms  (target < 50ms)")
        print(f"  Avg Energy           : {self.avg_energy():.6f} J")
        print(f"  Avg Total Cost       : {self.avg_cost():.6f}")
        print(f"  Task Completion Rate : {self.task_completion_rate():.1f}%  (target > 95%)")
        print(f"  Offload Ratio        : {self.offload_ratio():.1f}%")
        acc = self.offloading_accuracy()
        if acc is not None:
            print(f"  Offloading Accuracy  : {acc:.1f}%  (target > 90%)")
        print(f"{'='*55}")

    def to_dict(self):
        return {
            "strategy": self.strategy_name,
            "avg_latency_ms": self.avg_latency_ms(),
            "avg_energy": self.avg_energy(),
            "avg_cost": self.avg_cost(),
            "completion_rate": self.task_completion_rate(),
            "offload_ratio": self.offload_ratio(),
            "accuracy": self.offloading_accuracy(),
        }
