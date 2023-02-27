# D3G

## Overview
Distributed Differentiable Dynamic Game (D3G) is a framework, which enables learning multirobot coordination from demonstrations. We employ two types
of robots and two scenarios to validate Algorithms.
1. XEn.py uses a planar quadrotor model as robot dynamics, and define the cost function of different scenarios mentioned above.
2. GEn.py uses a unicyle model as robot dynamics, and define the cost function of different scenarios mentioned above.
3. GPDP.py implements the distributed game solver and the distributed differentiable Pontryagin’s Minimum Principle (PMP) solver algorithms.

## Usage

### Getting started

Download the source code folder and establish new projects in Pycharm (recommended). Make sure all associated files are properly linked to the project.

Requirements:

Python 3.10 tested

numpy 1.23.2 tested

casadi 3.5.5.post2 tested

matplotlib 3.5.3 tested

### The scenarios
The package contains three folers named as "". Each foler is corresponding to one specific sceario. In each folder, `generate_demo.py` is to generate a demonstration, and `formation_learning.py` is for learning from the demonstration. For each scenario, first run `generate_demo.py` to generate a dataset of demonstration which will be saved in the 'data' foler. Then run `formation_learning.py` to learn the demonstration.

## Simulation examples
