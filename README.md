# Multi-Armed Bandit (MAB) Simulation

This project provides a simulation of the multi-armed bandit (MAB) problem, a classic problem in probability theory, statistics, and machine learning. In the MAB problem, an agent must decide which arm of a set of arms (each with an unknown reward probability distribution) to pull in order to maximize cumulative reward over time.

## Overview

The simulation includes two popular algorithms for solving the MAB problem:

1. **Epsilon Greedy**: An algorithm that balances exploration and exploitation by choosing a random action with probability epsilon and the best action with probability (1 - epsilon).

2. **Thompson Sampling**: A probabilistic algorithm that maintains a distribution over the reward probabilities of each arm and selects actions based on sampling from these distributions.

## Contents

- `Bandit.py`: The main script to run the MAB simulation.
- `logs.py`: Custom logging configuration.
- `README.md`: This file, providing an overview and instructions.
- `requirements.txt`: Required Python packages.
- `LICENSE`: License information.

## Instructions

1. **Setup**: Install the required Python packages by running `pip install -r requirements.txt`.

2. **Run the Simulation**: Execute `Bandit.py` to run the MAB simulation. The script compares the performance of Epsilon Greedy and Thompson Sampling algorithms over a specified number of trials.

3. **Output**: The script produces visualizations comparing cumulative rewards and regrets of the two algorithms, as well as logging messages at different levels (debug, info, warning, error, critical).



