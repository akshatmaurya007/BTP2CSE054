"""
run_experiments.py
------------------
Main experiment runner. Trains the classifier, runs all scenarios,
compares proposed approach vs baselines, and saves result plots.

Scenarios:
  - Varying battery levels
  - Varying network congestion
  - Varying number of tasks
  - Varying task sizes

Usage:
  python run_experiments.py

Project: BTP2CSE054 - Task Offloading Decision System in MEC
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.task import Task
from simulator.mobile_device import MobileDevice
from simulator.edge_server import EdgeServer
from simulator.wireless_channel import WirelessChannel
from decision_engine.decision_engine import DecisionEngine
from evaluation.baselines import AlwaysLocal, AlwaysOffload, RandomBaseline, GreedyLatency
from evaluation.metrics import MetricsTracker

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_tasks.csv")


def load_and_train(engine):
    """Load dataset and train the ML classifier."""
    df = pd.read_csv(DATA_PATH)
    feature_cols = [
        "data_size", "cpu_cycles", "max_delay",
        "local_cpu_freq", "edge_cpu_freq",
        "uplink_rate", "battery_level", "network_latency"
    ]
    X = df[feature_cols].values
    y = df["label"].values
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    engine.train_classifier(X_train, y_train)
    engine.classifier.evaluate(X_test, y_test)
    engine.classifier.save()


def run_scenario(num_tasks, battery, congestion, engine, baselines):
    """Run one simulation scenario and return metrics for each strategy."""
    trackers = {b.name: MetricsTracker(b.name) for b in baselines}
    trackers["Proposed (LDROA-style)"] = MetricsTracker("Proposed (LDROA-style)")

    device  = MobileDevice(0, cpu_frequency=100e6, battery_level=battery)
    server  = EdgeServer(0, cpu_frequency=3e9)
    channel = WirelessChannel()
    channel.simulate_congestion(congestion)

    for i in range(num_tasks):
        task = Task.generate_random(task_id=i)

        # Proposed
        decision, _, metrics = engine.decide(task, device, channel, server)
        lat = metrics["local_latency_ms"] / 1000 if decision == "local" else (metrics.get("edge_latency_ms") or 0) / 1000
        eng = device.compute_local_energy(task) if decision == "local" else channel.transmission_energy(task)
        cost = metrics["local_cost"] if decision == "local" else (metrics.get("offload_cost") or 0)
        done = lat < task.max_delay
        trackers["Proposed (LDROA-style)"].record(lat, eng, cost, decision, done)

        # Baselines
        for b in baselines:
            dec, lat_b, eng_b, cost_b = b.decide(task, device, channel, server)
            done_b = lat_b < task.max_delay
            trackers[b.name].record(lat_b, eng_b, cost_b, dec, done_b)

    return trackers


def plot_comparison(results, x_values, x_label, metric_key, metric_label, filename):
    """Generic comparison plot across strategies."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for name, values in results.items():
        plt.plot(x_values, values, marker="o", label=name)
    plt.xlabel(x_label)
    plt.ylabel(metric_label)
    plt.title(f"{metric_label} vs {x_label}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    engine   = DecisionEngine(model_type="logistic")
    baselines = [AlwaysLocal(), AlwaysOffload(), RandomBaseline(), GreedyLatency()]

    # Train classifier
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "models", "trained_classifier.pkl")):
        print("[Main] Generating dataset first...")
        from data.generate_dataset import generate_dataset
        generate_dataset()
    load_and_train(engine)

    # ── Experiment 1: Vary number of tasks ──────────────────────────
    task_counts = [20, 40, 80, 120, 160, 200]
    lat_results  = {b.name: [] for b in baselines}
    lat_results["Proposed (LDROA-style)"] = []
    cost_results = {k: [] for k in lat_results}

    for n in task_counts:
        trackers = run_scenario(n, battery=80, congestion="none", engine=engine, baselines=baselines)
        for name, t in trackers.items():
            lat_results[name].append(t.avg_latency_ms())
            cost_results[name].append(t.avg_cost())

    plot_comparison(lat_results, task_counts, "Number of Tasks", "avg_latency",
                    "Avg Latency (ms)", "latency_vs_tasks.png")
    plot_comparison(cost_results, task_counts, "Number of Tasks", "avg_cost",
                    "Avg Total Cost", "cost_vs_tasks.png")

    # ── Experiment 2: Vary battery level ────────────────────────────
    battery_levels = [10, 20, 40, 60, 80, 100]
    lat_results2   = {b.name: [] for b in baselines}
    lat_results2["Proposed (LDROA-style)"] = []

    for bat in battery_levels:
        trackers = run_scenario(100, battery=bat, congestion="none", engine=engine, baselines=baselines)
        for name, t in trackers.items():
            lat_results2[name].append(t.avg_latency_ms())

    plot_comparison(lat_results2, battery_levels, "Battery Level (%)", "avg_latency",
                    "Avg Latency (ms)", "latency_vs_battery.png")

    # ── Experiment 3: Vary congestion ───────────────────────────────
    congestion_levels = ["none", "low", "moderate", "high"]
    lat_results3 = {b.name: [] for b in baselines}
    lat_results3["Proposed (LDROA-style)"] = []

    for cong in congestion_levels:
        trackers = run_scenario(100, battery=80, congestion=cong, engine=engine, baselines=baselines)
        for name, t in trackers.items():
            lat_results3[name].append(t.avg_latency_ms())

    plot_comparison(lat_results3, congestion_levels, "Congestion Level", "avg_latency",
                    "Avg Latency (ms)", "latency_vs_congestion.png")

    # Final summary
    print("\n[Main] All experiments complete. Figures saved to results/figures/")


if __name__ == "__main__":
    main()
