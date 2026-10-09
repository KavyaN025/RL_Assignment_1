# Reinforcement Learning Assignment 1 — Frozen Lake MDP

**Author:** Kavya Namburi  
**Roll Number:** BT23CSE004

## Project Overview

This project implements a 4×4 Frozen Lake Markov Decision Process (MDP) as part of Reinforcement Learning Assignment 1. It demonstrates policy evaluation, policy iteration, value iteration, exact policy evaluation using linear algebra, and experiments involving discount factors and living rewards.

The project includes a Python implementation, a PowerPoint presentation, a Word report, and a convergence plot.

## Project Structure

```text
RL_Assignment1_GitHub/
├── README.md
├── .gitignore
├── rl_assignment1.py
├── results_summary.txt
├── reports/
│   ├── RL_Assignment1_Presentation.pptx
│   └── RL_Assignment1_Report.docx
└── plots/
    └── RL_Assignment1_Convergence_Plot.png
```

## Environment Setup

The environment uses a 4×4 grid:

```text
S  H  F  F
F  F  F  H
H  F  F  F
F  F  H  G
```

- **S:** Starting state
- **F:** Frozen/safe cell
- **H:** Hole (terminal state)
- **G:** Goal (terminal state)

States are numbered from 0 to 15 in row-major order.

- Start state: `0`
- Hole states: `{1, 7, 8, 14}`
- Goal state: `15`
- Actions: Left, Down, Right, Up

### Rewards and Parameters

| Parameter | Value |
|---|---|
| Intended movement probability | 0.25 |
| Stay probability | 0.75 |
| Ordinary-step reward | -0.01 |
| Hole reward | -1 |
| Goal reward | +1 |
| Discount factor (γ) | 0.9 |
| Convergence tolerance | 1e-8 |

**Transition-model assumption:** The assignment specifies a probability of 0.25 for intended movement but does not fully specify the remaining probability. This implementation assumes the remaining 0.75 probability corresponds to staying in the current cell. If an intended move would cross a grid boundary, the agent remains in the current cell. This assumption is explicitly documented for reproducibility.

## Assignment Coverage

### Part 1 — MDP Formulation
- Definition of the MDP tuple: states, actions, transition probabilities, rewards, and discount factor.
- Explanation of whether the environment is finite, episodic, and fully observed.

### Part 2 — Policy Evaluation
- Uniform random policy with equal action probabilities.
- Bellman expectation equation and iterative policy evaluation.
- Manual calculation of the first two iterations for a selected state.
- Convergence analysis and plot.
- Calculation of the action-value function Qπ(s, a).

### Part 3 — Policy Iteration and Value Iteration
- Iterative policy evaluation and greedy policy improvement.
- Computation of the optimal value function and policy.
- Comparison of policy iteration and value iteration.
- Explanation of the Bellman operator's contraction property.

### Part 4 — Exact Policy Evaluation
- Formulation of policy evaluation as a linear system.
- Comparison between iterative evaluation and the exact solution.
- Discussion of the scalability limitations of dense linear algebra.

### Part 5 — Exploration
- Effect of changing the discount factor.
- Effect of changing the living reward from -0.01 to 0.
- Discussion of model-free methods when transition probabilities and rewards are unknown, including TD(0), Q-learning, and SARSA.

## Requirements

- Python 3
- NumPy
- Matplotlib

Install the required dependencies:

```bash
python3 -m pip install numpy matplotlib
```

## How to Run

Run the Python implementation from the project root:

```bash
python3 rl_assignment1.py
```

The script prints the calculated value functions, action values, policy iteration results, value iteration results, exact-solution comparison, and Part 5 experiment results.

It also generates a convergence plot named:

```text
RL_Assignment1_Convergence_Plot.png
```

The presentation and report are available in the `reports/` directory. The saved convergence plot is available in the `plots/` directory.

## Reproducibility

The implementation uses synchronous value updates, an initial value function of zero, a discount factor of 0.9 for the main experiment, and a convergence tolerance of 1e-8.

Expected results for the stated model include:

- Uniform-policy evaluation: 94 sweeps.
- Value iteration with γ = 0.9: 72 sweeps.
- Policy-iteration evaluation sweeps: 94 and 72.
- First manual check: V₁(2) = -0.071875.
- Second manual check: V₂(2) = -0.13251953125.

Results may differ if the transition model, reward convention, update method, or convergence tolerance is changed.

## How to Push the Project to GitHub

### 1. Create a GitHub Repository

Create a new repository at [GitHub](https://github.com/new), for example:

`rl-assignment-1-frozen-lake`

For the simplest setup, create the repository without initializing it with another README.

### 2. Open a Terminal in the Project Folder

Navigate to the extracted project directory:

```bash
cd path/to/RL_Assignment1_GitHub
```

Replace the example path with the actual location of your folder.

### 3. Initialize Git and Commit the Files

```bash
git init
git add .
git status
git commit -m "Add RL Assignment 1 project files"
```

Check the output of `git status` before committing to ensure you are adding the intended files.

If Git asks you to configure your identity, run:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

Use your own name and email, then repeat the commit command.

### 4. Connect to GitHub and Push

Replace the example URL with your actual repository URL:

```bash
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/rl-assignment-1-frozen-lake.git
git push -u origin main
```

Authenticate with GitHub if prompted. Never commit passwords, access tokens, or other secrets.

### 5. Verify the Upload

Refresh your GitHub repository page and check that the README, Python implementation, presentation, Word report, convergence plot, and results summary are present.

## Notes

- Confirm the transition-model assumption with your instructor before submission.
- Run the Python implementation to reproduce the numerical findings.
- Review the report and presentation against the assignment requirements.
- Follow your institution's rules before making assignment solutions publicly accessible.
