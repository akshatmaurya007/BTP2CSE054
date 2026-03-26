"""
mec_environment.py
------------------
Ties together MobileDevice, EdgeServer, WirelessChannel, and Task
into a unified simulation environment.

Project: BTP2CSE054 - Task Offloading Decision System in MEC
"""

from simulator.mobile_device import MobileDevice
from simulator.edge_server import EdgeServer
from simulator.wireless_channel import WirelessChannel
from simulator.task import Task


class MECEnvironment:
    """
    Simulates a MEC environment with one mobile device,
    multiple edge servers, and a wireless channel.
    """

    def __init__(self, num_edge_servers=3, device_cpu_mhz=100, battery=100.0):
        self.device = MobileDevice(
            device_id=0,
            cpu_frequency=device_cpu_mhz * 1e6,
            battery_level=battery
        )
        self.edge_servers = [
            EdgeServer(server_id=i, cpu_frequency=(1 + i) * 1e9)
            for i in range(num_edge_servers)
        ]
        self.channel = WirelessChannel()
        self.task_counter = 0

    def get_best_edge_server(self):
        """Select the edge server with the most available capacity."""
        available = [s for s in self.edge_servers if s.utilization() < 80]
        if not available:
            return None
        return max(available, key=lambda s: s.cpu_frequency - s.current_load)

    def generate_task(self):
        """Generate a new random task."""
        task = Task.generate_random(task_id=self.task_counter)
        self.task_counter += 1
        return task

    def compute_offload_total_latency(self, task, server):
        """Total offloading latency = transmission time + edge execution time."""
        t_off = self.channel.transmission_latency(task)
        t_edge = server.compute_edge_latency(task)
        return t_off + t_edge

    def compute_offload_total_cost(self, task, server):
        """Total offloading cost = offloading cost + edge execution cost."""
        t_off = self.channel.transmission_latency(task)
        e_off = self.channel.transmission_energy(task)
        t_edge = server.compute_edge_latency(task)
        cost_off = t_off + e_off
        cost_total = cost_off + t_edge
        return cost_total

    def set_congestion(self, level="none"):
        """Apply congestion to the wireless channel."""
        self.channel.simulate_congestion(level)

    def reset(self):
        """Reset the environment for a new simulation run."""
        for server in self.edge_servers:
            server.current_load = 0.0
        self.task_counter = 0

    def status(self):
        print(f"Device   : {self.device}")
        print(f"Channel  : {self.channel}")
        for s in self.edge_servers:
            print(f"Server   : {s}")
