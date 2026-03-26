"""
mobile_device.py
----------------
Models a mobile device (MD) with local CPU and battery constraints.

Project: BTP2CSE054 - Task Offloading Decision System in MEC
"""


class MobileDevice:
    """
    Represents a mobile device that generates tasks and executes them locally.

    Attributes:
        device_id       : Unique identifier
        cpu_frequency   : Local CPU speed in Hz (f_local)
        battery_level   : Remaining battery percentage (0-100)
        capacitance     : Effective switched capacitance (chip architecture constant)
    """

    def __init__(self, device_id, cpu_frequency=1e8, battery_level=100.0, capacitance=1e-28):
        self.device_id = device_id
        self.cpu_frequency = cpu_frequency      # Hz, e.g. 100 MHz
        self.battery_level = battery_level      # percentage
        self.capacitance = capacitance          # chip capacitance constant (alpha)

    def compute_local_latency(self, task):
        """
        Calculate time to execute task locally.
        t_local = C / f_local  (seconds)
        """
        return task.cpu_cycles / self.cpu_frequency

    def compute_local_energy(self, task):
        """
        Calculate energy consumed for local execution.
        e_local = alpha * C * f_local^2  (Joules)
        """
        return self.capacitance * task.cpu_cycles * (self.cpu_frequency ** 2)

    def compute_local_cost(self, task):
        """Total cost = latency + energy (combined metric)."""
        latency = self.compute_local_latency(task)
        energy = self.compute_local_energy(task)
        return latency + energy

    def drain_battery(self, energy_joules, drain_rate=0.001):
        """Simulate battery drain proportional to energy consumed."""
        self.battery_level = max(0.0, self.battery_level - energy_joules * drain_rate)

    def is_battery_critical(self, threshold=15.0):
        """Returns True if battery is below critical threshold."""
        return self.battery_level < threshold

    def __repr__(self):
        return (
            f"MobileDevice(id={self.device_id}, "
            f"cpu={self.cpu_frequency/1e6:.0f}MHz, "
            f"battery={self.battery_level:.1f}%)"
        )
