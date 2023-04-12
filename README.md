# D3G

## Overview
Distributed Differentiable Dynamic Game (D3G) is a framework, which enables learning multirobot coordination from demonstrations. We employ two types
of robots and two scenarios to validate Algorithms.
1. XEnv.py uses a planar quadrotor model as robot dynamics, and define the cost function of different scenarios mentioned above.
2. GEnv.py uses a unicyle model as robot dynamics, and define the cost function of different scenarios mentioned above.
3. GPDP.py implements the distributed game solver and the distributed differentiable Pontryagin’s Minimum Principle (PMP) solver algorithms for unicyle model.
4. GPDP.py implements the distributed game solver and the distributed differentiable Pontryagin’s Minimum Principle (PMP) solver algorithms for quadrotor planner.

## Usage

### Getting started

Download the source code folder and establish new projects in Pycharm (recommended). Make sure all associated files are properly linked to the project.

Requirements:

Python 3.10 tested

numpy 1.23.2 tested

casadi 3.5.5.post2 tested

matplotlib 3.5.3 tested

### The scenarios
The package contains five folers named as "uavswarm2D_diamond", "uavswarm2D_line", "uavswarm2D_transition", "uavswarm3D_quadrotor", and "uavswarm_unicyle". Each foler is corresponding to one specific sceario. In each folder, `generate_demo.py` is to generate a demonstration, and `formation_learning.py` is for learning from the demonstration. For each scenario, first run `generate_demo.py` to generate a dataset of demonstration which will be saved in the 'data' foler. Then run `formation_learning.py` to learn the demonstration.

## Simulation examples
1)  uavswarm2D_line: 
the robots start from different initial positions with 0 speed, their goal is to initialize,
within the time horizon, a desired (linear-like) formation towards the Y axis, with velocity 2.
During this process, they have to avoid collision and risky areas and walls:
![F1](https://user-images.githubusercontent.com/71677216/231520433-2df07d13-4d7b-4b20-a88b-ad172187bb21.jpg)



2) uavswarm2D_transition:
Similar to exmaple (1), but the robots aim to initialize a diamond formation:
![F2](https://user-images.githubusercontent.com/71677216/231520467-c0099fbe-39a9-4e31-b749-1d3db90a45f0.jpg)

3) uavswarm2D_transiition:
The robots are initialized with a diamond formation obtained in exmaple (2). Robots are aware of a potential target from in direction of F , they want to form a
new formation offering them positional advantage against that target.
![F3](https://user-images.githubusercontent.com/71677216/231520934-bd59e1ad-fdbb-43dc-902d-87c3fb78946e.jpg)




