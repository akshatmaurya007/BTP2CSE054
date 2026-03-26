# BTP2CSE054 — Task Offloading Decision System in MEC

> **IILM University, Greater Noida | School of Computer Science and Engineering**
> Minor Project | Session 2025-26 | SDG Goal 9 — Industry, Innovation and Infrastructure

---

## Team

| Name | Roll No |
|------|---------|
| Akshat Maurya | 2410030648 |
| Ritik Jaiswal | U910030619 |
| Priyanshu Singh | U910030667 |
| Vanshika Aggarwal | 2410030639 |
| Ayush Patel | 2410030614 |

**Guide:** Dr. Prakhar

---

## Problem Statement

Mobile devices running heavy applications suffer from high execution delay and excessive energy consumption due to limited CPU and battery capacity. While Mobile Edge Computing (MEC) provides nearby computing support via edge servers, improper offloading decisions can paradoxically *increase* latency and energy rather than reducing them.

This project builds an **intelligent, adaptive Task Offloading Decision System** that dynamically evaluates device state, network conditions, and task characteristics to make optimal offloading decisions — minimising execution latency and energy consumption in real time.

---

## System Architecture

```
Mobile Device
    │
    ▼
┌─────────────────────────────────┐
│         Decision Engine          │
│                                  │
│  Stage 1: Rule-Based Pre-filter  │──► LOCAL
│      ↓ (uncertain cases)         │
│  Stage 2: ML Classifier          │──► LOCAL / OFFLOAD
└─────────────────────────────────┘
         │ offload
         ▼
   Wireless Channel (Shannon model)
         │
         ▼
   MEC Edge Server (up to 3)
         │
         ▼
   Result returned to device
```

---

## Repository Structure

```
BTP2CSE054/
│
├── simulator/                  # MEC environment components
│   ├── task.py                 # Task model (data size, CPU cycles, deadline)
│   ├── mobile_device.py        # Mobile device (CPU, battery, local execution)
│   ├── edge_server.py          # MEC server (CPU, capacity, edge execution)
│   ├── wireless_channel.py     # Shannon uplink model
│   └── mec_environment.py      # Full simulation environment
│
├── decision_engine/            # Core decision-making system
│   ├── rule_based_filter.py    # Stage 1: hard threshold rules
│   ├── ml_classifier.py        # Stage 2: Logistic Regression / MLP
│   └── decision_engine.py      # Combines both stages
│
├── data/
│   └── generate_dataset.py     # Synthetic dataset generator (5000 samples)
│
├── models/                     # Saved trained models (generated at runtime)
│
├── evaluation/
│   ├── baselines.py            # Always-Local, Always-Offload, Random, Greedy
│   ├── metrics.py              # Latency, energy, accuracy, completion rate
│   └── run_experiments.py      # Main experiment runner + plot generator
│
├── results/figures/            # Output plots (generated at runtime)
│
├── docs/
│   └── synopsis.docx           # Project synopsis document
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/akshatmaurya007/BTP2CSE054.git
cd BTP2CSE054
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate synthetic dataset
```bash
python data/generate_dataset.py
```

### 4. Run all experiments
```bash
python evaluation/run_experiments.py
```

Results and plots will appear in `results/figures/`.

---

## Decision Logic

### Stage 1 — Rule-Based Pre-filter
| Condition | Action |
|-----------|--------|
| Battery < 15% | Force LOCAL |
| Network latency > 50% of task deadline | Force LOCAL |
| Data size < 10 KB | Force LOCAL |
| None of the above | Pass to Stage 2 |

### Stage 2 — ML Classifier
A **Logistic Regression** (or lightweight MLP) trained on 5,000 synthetic samples.

**Input features:** data size, CPU cycles, deadline, local CPU freq, edge CPU freq, uplink rate, battery level, network latency

**Output:** 0 = execute locally | 1 = offload to edge server

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Avg Task Latency | < 50 ms |
| Offloading Accuracy | > 90% |
| Task Completion Rate | > 95% |
| Edge CPU Utilisation | 40–70% |

---

## Baseline Comparisons

- **Always-Local** — never offloads
- **Always-Offload** — always offloads
- **Random** — 50/50 random choice
- **Greedy-Latency** — picks lower latency, ignores energy

---

## Technologies

| Category | Tool |
|----------|------|
| Language | Python 3.x |
| ML Framework | Scikit-learn / TensorFlow |
| Simulation | Custom Python MEC Simulator |
| Data Analysis | NumPy, Pandas, Matplotlib |
| Version Control | GitHub |

---

## References

Based on the research paper:
> Eang et al., *"Offloading Decision and Resource Allocation in Mobile Edge Computing for Cost and Latency Efficiencies in Real-Time IoT"*, Electronics 2024, 13, 1218.
> https://doi.org/10.3390/electronics13071218
