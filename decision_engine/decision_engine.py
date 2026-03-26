"""
decision_engine.py
------------------
The core Decision Engine (DE) for the Task Offloading Decision System.

Two-stage hybrid approach:
  Stage 1 — Rule-Based Pre-filter  : handles obvious cases instantly
  Stage 2 — ML Classifier          : handles uncertain/borderline cases

Final output for each task:
  'local'   → execute on mobile device
  'offload' → send to MEC edge server

Project: BTP2CSE054 - Task Offloading Decision System in MEC
"""

from decision_engine.rule_based_filter import RuleBasedFilter
from decision_engine.ml_classifier import OffloadingClassifier, FEATURE_NAMES


class DecisionEngine:
    """
    Combines rule-based filtering and ML classification to make
    optimal task offloading decisions in real time.
    """

    def __init__(self, model_type="logistic"):
        self.rule_filter = RuleBasedFilter()
        self.classifier = OffloadingClassifier(model_type=model_type)
        self.decision_log = []

    def train_classifier(self, X_train, y_train):
        """Train the ML classifier with synthetic or real dataset."""
        self.classifier.train(X_train, y_train)

    def load_classifier(self):
        """Load a pre-trained classifier from disk."""
        self.classifier.load()

    def decide(self, task, device, channel, edge_server):
        """
        Make an offloading decision for a given task.

        Returns:
            decision : str — 'local' or 'offload'
            reason   : str — explanation of the decision
            metrics  : dict — computed cost/latency values
        """
        net_latency = channel.network_latency()

        # ── Stage 1: Rule-Based Filter ──────────────────────────────────
        stage1_decision, stage1_reason = self.rule_filter.decide(
            task, device, net_latency
        )

        if stage1_decision == "local":
            local_latency = device.compute_local_latency(task)
            local_cost    = device.compute_local_cost(task)
            metrics = {
                "stage": 1,
                "local_latency_ms": local_latency * 1000,
                "local_cost": local_cost,
                "edge_latency_ms": None,
                "offload_cost": None,
                "net_latency_ms": net_latency * 1000,
            }
            self._log(task, "local", stage1_reason, metrics)
            return "local", stage1_reason, metrics

        # ── Stage 2: ML Classifier ──────────────────────────────────────
        uplink_rate = channel.uplink_rate()

        features = [
            task.data_size,
            task.cpu_cycles,
            task.max_delay,
            device.cpu_frequency,
            edge_server.cpu_frequency,
            uplink_rate,
            device.battery_level,
            net_latency,
        ]

        ml_pred = self.classifier.predict(features)
        decision = "offload" if ml_pred == 1 else "local"

        # Compute actual costs for logging
        local_latency  = device.compute_local_latency(task)
        local_cost     = device.compute_local_cost(task)
        t_off          = channel.transmission_latency(task)
        t_edge         = edge_server.compute_edge_latency(task)
        offload_cost   = channel.transmission_energy(task) + t_off + t_edge

        metrics = {
            "stage": 2,
            "local_latency_ms": local_latency * 1000,
            "local_cost": local_cost,
            "edge_latency_ms": (t_off + t_edge) * 1000,
            "offload_cost": offload_cost,
            "net_latency_ms": net_latency * 1000,
        }

        reason = (
            f"ML classifier ({self.classifier.model_type}) predicted "
            f"{'offloading' if ml_pred == 1 else 'local'} is better"
        )
        self._log(task, decision, reason, metrics)
        return decision, reason, metrics

    def _log(self, task, decision, reason, metrics):
        self.decision_log.append({
            "task_id": task.task_id,
            "decision": decision,
            "reason": reason,
            **metrics
        })

    def summary(self):
        """Print a summary of all decisions made so far."""
        total = len(self.decision_log)
        offloaded = sum(1 for d in self.decision_log if d["decision"] == "offload")
        local = total - offloaded
        print(f"\n{'='*50}")
        print(f"Decision Engine Summary ({total} tasks)")
        print(f"  Offloaded : {offloaded} ({100*offloaded/max(total,1):.1f}%)")
        print(f"  Local     : {local} ({100*local/max(total,1):.1f}%)")
        print(f"{'='*50}")
