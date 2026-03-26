"""
edge_server.py
--------------
Models a MEC (Mobile Edge Computing) server at the network boundary.

Project: BTP2CSE054 - Task Offloading Decision System in MEC
"""


class EdgeServer:
    """
    Represents a MEC edge server co-located with a wireless access point.

    Attributes:
        server_id       : Unique identifier
        cpu_frequency   : Edge server CPU speed in Hz (f_edge)
        max_capacity    : Maximum total CPU cycles it can handle simultaneously
        capacitance     : Effective capacitance for energy model
    """

    def __init__(self, server_id, cpu_frequency=3e9, max_capacity=1e12, capacitance=1e-27):
        self.server_id = server_id
        self.cpu_frequency = cpu_frequency      # Hz, e.g. 3 GHz
        self.max_capacity = max_capacity        # total cycles supported
        self.capacitance = capacitance
        self.current_load = 0.0                 # cycles currently in use

    def compute_edge_latency(self, task):
        """
        Execution time at edge server.
        t_edge = C / f_edge  (seconds)
        """
        return task.cpu_cycles / self.cpu_frequency

    def compute_edge_energy(self, task):
        """
        Energy consumed by edge server to execute task.
        e_edge = alpha * C * f_edge^2
        (Note: edge server is grid-powered, so this is less critical)
        """
        return self.capacitance * task.cpu_cycles * (self.cpu_frequency ** 2)

    def compute_edge_cost(self, task):
        """Edge execution cost = edge latency + edge energy."""
        return self.compute_edge_latency(task) + self.compute_edge_energy(task)

    def has_capacity(self, task):
        """Check if the server has enough remaining capacity for this task."""
        return (self.current_load + task.cpu_cycles) <= self.max_capacity

    def allocate(self, task):
        """Reserve resources for a task."""
        if self.has_capacity(task):
            self.current_load += task.cpu_cycles
            return True
        return False

    def release(self, task):
        """Free resources after task completes."""
        self.current_load = max(0, self.current_load - task.cpu_cycles)

    def utilization(self):
        """Returns current CPU utilization as a percentage."""
        return (self.current_load / self.max_capacity) * 100

    def __repr__(self):
        return (
            f"EdgeServer(id={self.server_id}, "
            f"cpu={self.cpu_frequency/1e9:.1f}GHz, "
            f"load={self.utilization():.1f}%)"
        )
