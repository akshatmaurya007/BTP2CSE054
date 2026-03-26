"""
rule_based_filter.py
--------------------
Stage 1 of the Decision Engine.
Applies hard threshold rules to immediately decide obvious cases
before passing ambiguous tasks to the ML classifier.

Rules (from project synopsis):
  1. Battery < 15%           → Force LOCAL (preserve battery)
  2. Network latency > τ/2   → Force LOCAL (offloading too slow)
  3. Data size very small     → Force LOCAL (offloading overhead not justified)

Project: BTP2CSE054 - Task Offloading Decision System in MEC
"""

SMALL_DATA_THRESHOLD_BITS = 1e4    # 10 KB — below this, offloading overhead dominates
BATTERY_CRITICAL_PCT      = 15.0   # force local below this battery level
LATENCY_RATIO             = 0.5    # if net latency > 50% of deadline, force local


class RuleBasedFilter:
    """
    Applies deterministic threshold rules for fast, obvious offloading decisions.

    Returns:
        'local'    — task must run on device
        'offload'  — task must be offloaded (no hard rules triggered this)
        'uncertain' — cannot decide; pass to ML classifier
    """

    def __init__(
        self,
        battery_threshold=BATTERY_CRITICAL_PCT,
        latency_ratio=LATENCY_RATIO,
        small_data_threshold=SMALL_DATA_THRESHOLD_BITS
    ):
        self.battery_threshold = battery_threshold
        self.latency_ratio = latency_ratio
        self.small_data_threshold = small_data_threshold

    def decide(self, task, device, network_latency):
        """
        Apply rules in order. Returns ('local'|'uncertain', reason_string).

        Parameters:
            task            : Task object
            device          : MobileDevice object
            network_latency : Current measured round-trip latency (seconds)
        """
        # Rule 1: Critical battery → always local
        if device.battery_level < self.battery_threshold:
            return "local", f"Battery critical ({device.battery_level:.1f}% < {self.battery_threshold}%)"

        # Rule 2: Network too slow relative to task deadline
        if network_latency > self.latency_ratio * task.max_delay:
            return "local", (
                f"Network latency ({network_latency*1000:.1f}ms) exceeds "
                f"{int(self.latency_ratio*100)}% of deadline ({task.max_delay*1000:.1f}ms)"
            )

        # Rule 3: Tiny data size — offloading overhead not worth it
        if task.data_size < self.small_data_threshold:
            return "local", f"Data size too small ({task.data_size/1e3:.1f}KB < {self.small_data_threshold/1e3:.0f}KB)"

        # No hard rule triggered — pass to ML classifier
        return "uncertain", "No hard rule triggered — defer to ML classifier"

    def __repr__(self):
        return (
            f"RuleBasedFilter(battery_thresh={self.battery_threshold}%, "
            f"latency_ratio={self.latency_ratio}, "
            f"min_data={self.small_data_threshold/1e3:.0f}KB)"
        )
