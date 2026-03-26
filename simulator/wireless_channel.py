"""
wireless_channel.py
-------------------
Models the wireless uplink channel between mobile device and edge server.
Uses Shannon's channel capacity theorem for realistic data rate estimation.

Project: BTP2CSE054 - Task Offloading Decision System in MEC
"""

import math
import random


class WirelessChannel:
    """
    Models the wireless uplink channel (device → edge server).

    Attributes:
        bandwidth       : System bandwidth in Hz (W_nj)
        tx_power        : Transmission power of device in Watts (p_tr)
        channel_gain    : Power gain between UE and base station (g_nj)
        noise_power     : Background noise power in Watts (n_0)
        intercell_interference : Inter-cell interference (L_os)
    """

    def __init__(
        self,
        bandwidth=10e6,             # 10 MHz
        tx_power=0.2,               # 200 mW
        channel_gain=0.1,           # power gain
        noise_power=1e-10,          # thermal noise
        intercell_interference=0.0  # assumed negligible in single-cell
    ):
        self.bandwidth = bandwidth
        self.tx_power = tx_power
        self.channel_gain = channel_gain
        self.noise_power = noise_power
        self.intercell_interference = intercell_interference

    def uplink_rate(self):
        """
        Shannon capacity formula:
        R = W * log2(1 + (p * g) / (n0 + interference))
        Returns data rate in bits/second.
        """
        sinr = (self.tx_power * self.channel_gain) / (
            self.noise_power + self.intercell_interference
        )
        return self.bandwidth * math.log2(1 + sinr)

    def transmission_latency(self, task):
        """
        Time to transmit task data to edge server.
        t_off = D / R  (seconds)
        """
        rate = self.uplink_rate()
        return task.data_size / rate

    def transmission_energy(self, task):
        """
        Energy consumed during uplink transmission.
        e_off = p_tr * t_off  (Joules)
        """
        t_off = self.transmission_latency(task)
        return self.tx_power * t_off

    def network_latency(self):
        """
        Round-trip network latency (simulated).
        Returns latency in seconds.
        """
        # Simulates realistic variation between 5ms and 50ms
        return random.uniform(0.005, 0.05)

    def simulate_congestion(self, congestion_level="none"):
        """
        Adjusts channel gain to simulate network congestion.
        congestion_level: 'none', 'low', 'moderate', 'high'
        """
        factors = {"none": 1.0, "low": 0.7, "moderate": 0.4, "high": 0.1}
        factor = factors.get(congestion_level, 1.0)
        self.channel_gain *= factor

    def __repr__(self):
        return (
            f"WirelessChannel(bw={self.bandwidth/1e6:.0f}MHz, "
            f"rate={self.uplink_rate()/1e6:.1f}Mbps)"
        )
