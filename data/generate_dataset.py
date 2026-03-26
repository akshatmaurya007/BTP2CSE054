"""
generate_dataset.py
-------------------
Generates a synthetic dataset of MEC tasks with ground-truth
offloading labels derived from the analytical cost model.

Label logic:
  offload (1) if:  offload_total_cost < local_cost
                   AND (t_off + t_edge) < task.max_delay
  else local (0)

Output: data/synthetic_tasks.csv

Project: BTP2CSE054 - Task Offloading Decision System in MEC
"""

import os
import random
import numpy as np
import pandas as pd

from simulator.task import Task
from simulator.mobile_device import MobileDevice
from simulator.edge_server import EdgeServer
from simulator.wireless_channel import WirelessChannel

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "synthetic_tasks.csv")
NUM_SAMPLES = 5000
RANDOM_SEED = 42


def generate_dataset(n=NUM_SAMPLES, seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)

    records = []

    for i in range(n):
        # Randomly sample environment parameters
        local_cpu  = random.uniform(10e6, 200e6)     # 10–200 MHz
        edge_cpu   = random.uniform(1e9, 4e9)        # 1–4 GHz
        battery    = random.uniform(5, 100)           # 5–100%
        bandwidth  = random.uniform(5e6, 20e6)       # 5–20 MHz
        tx_power   = random.uniform(0.1, 0.5)        # 100–500 mW
        ch_gain    = random.uniform(0.01, 0.5)
        noise      = 1e-10

        task = Task.generate_random(task_id=i)
        device = MobileDevice(0, cpu_frequency=local_cpu, battery_level=battery)
        server = EdgeServer(0, cpu_frequency=edge_cpu)
        channel = WirelessChannel(bandwidth, tx_power, ch_gain, noise)

        # Compute costs analytically
        local_latency  = device.compute_local_latency(task)
        local_energy   = device.compute_local_energy(task)
        local_cost     = local_latency + local_energy

        t_off         = channel.transmission_latency(task)
        e_off         = channel.transmission_energy(task)
        t_edge        = server.compute_edge_latency(task)
        offload_cost  = t_off + e_off + t_edge

        uplink_rate   = channel.uplink_rate()
        net_latency   = random.uniform(0.005, 0.05)

        # Ground-truth label
        label = int(
            offload_cost < local_cost and
            (t_off + t_edge) < task.max_delay
        )

        records.append({
            "data_size": task.data_size,
            "cpu_cycles": task.cpu_cycles,
            "max_delay": task.max_delay,
            "local_cpu_freq": local_cpu,
            "edge_cpu_freq": edge_cpu,
            "uplink_rate": uplink_rate,
            "battery_level": battery,
            "network_latency": net_latency,
            "local_cost": local_cost,
            "offload_cost": offload_cost,
            "label": label,           # 0=local, 1=offload
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[Dataset] Saved {n} samples to {OUTPUT_PATH}")
    print(f"[Dataset] Offload ratio: {df['label'].mean()*100:.1f}%")
    return df


if __name__ == "__main__":
    generate_dataset()
