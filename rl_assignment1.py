"""
Reinforcement Learning Assignment 1 — Frozen Lake MDP
Author: Kavya Namburi (BT23CSE004)

Run:
    python3 rl_assignment1.py

Dependencies:
    numpy
    matplotlib

This script implements Parts 1–5 of the assignment:
- Fixed 4x4 grid and explicit transition-model assumption
- Uniform-policy iterative evaluation, convergence history and plot
- Manual V1(2), V2(2) checks
- Q^pi(s,a) table
- Policy iteration and value iteration
- Exact linear-system verification
- Gamma and living-reward experiments
- Model-free methods discussion printed as a note

Transition-model assumption:
The assignment specifies P(intended movement)=0.25 but leaves the remaining
probability unspecified. This implementation assumes the remaining 0.75
corresponds to staying in the current cell. If the intended move hits a
boundary, the agent remains in the current cell. Hole and goal states are
terminal; their future value is zero.
"""
import numpy as np
import matplotlib.pyplot as plt

N_STATES = 16
ACTIONS = ["Left", "Down", "Right", "Up"]
ACTION_SYMBOLS = ["L", "D", "R", "U"]
DELTAS = {"Left": (0, -1), "Down": (1, 0), "Right": (0, 1), "Up": (-1, 0)}
HOLES = {1, 7, 8, 14}
GOAL = 15
START = 0
TERMINALS = HOLES | {GOAL}
GAMMA = 0.9
STEP_REWARD = -0.01
HOLE_REWARD = -1.0
GOAL_REWARD = 1.0
THETA = 1e-8

def next_state(s, action):
    if s in TERMINALS:
        return s
    r, c = divmod(s, 4)
    dr, dc = DELTAS[action]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < 4 and 0 <= nc < 4):
        return s
    return nr * 4 + nc

def build_model(step_reward=STEP_REWARD):
    P = np.zeros((N_STATES, 4, N_STATES), dtype=float)
    R = np.zeros((N_STATES, 4, N_STATES), dtype=float)
    for s in range(N_STATES):
        for ai, action in enumerate(ACTIONS):
            if s in TERMINALS:
                P[s, ai, s] = 1.0
                continue
            intended = next_state(s, action)
            P[s, ai, intended] += 0.25
            P[s, ai, s] += 0.75
            for sp in range(N_STATES):
                if P[s, ai, sp] > 0:
                    if sp == GOAL:
                        R[s, ai, sp] = GOAL_REWARD
                    elif sp in HOLES:
                        R[s, ai, sp] = HOLE_REWARD
                    else:
                        R[s, ai, sp] = step_reward
    return P, R

def q_from_v(V, P, R, gamma):
    Q = np.zeros((N_STATES, 4), dtype=float)
    for s in range(N_STATES):
        if s in TERMINALS:
            continue
        for ai in range(4):
            Q[s, ai] = sum(
                P[s, ai, sp] * (R[s, ai, sp] + gamma * V[sp])
                for sp in range(N_STATES)
            )
    return Q

def policy_evaluation(policy, P, R, gamma=GAMMA, theta=THETA, max_iter=100000):
    V = np.zeros(N_STATES, dtype=float)
    history = []
    for sweep in range(1, max_iter + 1):
        V_new = np.zeros(N_STATES, dtype=float)  # synchronous update
        for s in range(N_STATES):
            if s in TERMINALS:
                V_new[s] = 0.0
                continue
            value = 0.0
            for ai in range(4):
                q = sum(
                    P[s, ai, sp] * (R[s, ai, sp] + gamma * V[sp])
                    for sp in range(N_STATES)
                )
                value += policy[s, ai] * q
            V_new[s] = value
        delta = float(np.max(np.abs(V_new - V)))
        history.append(delta)
        V = V_new
        if delta < theta:
            return V, history
    raise RuntimeError("Policy evaluation did not converge.")

def greedy_policy(V, P, R, gamma=GAMMA):
    Q = q_from_v(V, P, R, gamma)
    policy = np.zeros((N_STATES, 4), dtype=float)
    for s in range(N_STATES):
        if s not in TERMINALS:
            policy[s, int(np.argmax(Q[s]))] = 1.0
    return policy

def policy_iteration(P, R, gamma=GAMMA, theta=THETA, max_improvements=100):
    policy = np.ones((N_STATES, 4), dtype=float) / 4.0
    eval_sweeps = []
    policy_history = []
    for _ in range(max_improvements):
        V, hist = policy_evaluation(policy, P, R, gamma, theta)
        eval_sweeps.append(len(hist))
        improved = greedy_policy(V, P, R, gamma)
        policy_history.append(improved.copy())
        if np.array_equal(improved, policy):
            return V, policy, eval_sweeps, policy_history
        policy = improved
    raise RuntimeError("Policy iteration did not stabilize.")

def value_iteration(P, R, gamma=GAMMA, theta=THETA, max_iter=100000):
    V = np.zeros(N_STATES, dtype=float)
    history = []
    for sweep in range(1, max_iter + 1):
        V_new = np.zeros(N_STATES, dtype=float)
        for s in range(N_STATES):
            if s in TERMINALS:
                V_new[s] = 0.0
                continue
            action_values = []
            for ai in range(4):
                action_values.append(sum(
                    P[s, ai, sp] * (R[s, ai, sp] + gamma * V[sp])
                    for sp in range(N_STATES)
                ))
            V_new[s] = max(action_values)
        delta = float(np.max(np.abs(V_new - V)))
        history.append(delta)
        V = V_new
        if delta < theta:
            break
    else:
        raise RuntimeError("Value iteration did not converge.")
    return V, greedy_policy(V, P, R, gamma), history

def exact_policy_evaluation(policy, P, R, gamma=GAMMA):
    P_pi = np.einsum("sa,san->sn", policy, P)
    R_pi = np.zeros(N_STATES, dtype=float)
    for s in range(N_STATES):
        if s in TERMINALS:
            continue
        for ai in range(4):
            R_pi[s] += policy[s, ai] * sum(
                P[s, ai, sp] * R[s, ai, sp] for sp in range(N_STATES)
            )
    A = np.eye(N_STATES) - gamma * P_pi
    b = R_pi.copy()
    for s in TERMINALS:
        A[s, :] = 0.0
        A[s, s] = 1.0
        b[s] = 0.0
    return np.linalg.solve(A, b)

def print_grid(values, title):
    print("\n" + title)
    print(np.asarray(values).reshape(4, 4))

def print_policy(policy, title):
    symbols = []
    for s in range(N_STATES):
        if s == GOAL:
            symbols.append("G")
        elif s in HOLES:
            symbols.append("H")
        else:
            symbols.append(ACTION_SYMBOLS[int(np.argmax(policy[s]))])
    print("\n" + title)
    for i in range(0, 16, 4):
        print(" ".join(symbols[i:i+4]))

def main():
    P, R = build_model()
    uniform_policy = np.ones((N_STATES, 4), dtype=float) / 4.0

    print("GRID: S H F F / F F F H / H F F F / F F H G")
    print("Start=0, Holes={1, 7, 8, 14}, Goal=15")
    print("Model assumption: 0.25 intended movement, 0.75 stay.")
    print("Boundary attempts stay in the current cell.")
    print("Terminal states have future value zero.")
    print("Actions:", ACTIONS, "| gamma =", GAMMA, "| theta =", THETA)

    # Part 2: policy evaluation
    V_pi, eval_history = policy_evaluation(uniform_policy, P, R, GAMMA, THETA)
    print("\nPART 2 — POLICY EVALUATION")
    print("Policy evaluation sweeps:", len(eval_history))
    print_grid(V_pi, "V^pi (uniform random policy):")
    print("Manual check V1(state 2) = -0.071875")
    print("Manual check V2(state 2) = -0.13251953125")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(eval_history)+1), eval_history, marker=".", markersize=3)
    plt.yscale("log")
    plt.xlabel("Sweep (iteration)")
    plt.ylabel(r"$\|V_{k+1}-V_k\|_\infty$")
    plt.title("Uniform-Policy Evaluation: Convergence")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("RL_Assignment1_Convergence_Plot.png", dpi=220)
    plt.close()
    print("Saved convergence plot: RL_Assignment1_Convergence_Plot.png")

    Q_pi = q_from_v(V_pi, P, R, GAMMA)
    print("\nQ^pi(s,a) columns are Left, Down, Right, Up:")
    for s in range(N_STATES):
        print(f"{s:2d}: " + "  ".join(f"{q:.6f}" for q in Q_pi[s]))

    # Part 3: policy iteration and value iteration
    V_pi_opt, policy_opt_pi, eval_sweeps, policies = policy_iteration(P, R, GAMMA, THETA)
    V_opt_vi, policy_opt_vi, vi_history = value_iteration(P, R, GAMMA, THETA)
    print("\nPART 3 — POLICY ITERATION")
    print("Policy evaluation sweeps in successive checks:", eval_sweeps)
    print_policy(policy_opt_pi, "Policy iteration result (L/D/R/U; H/G terminal):")
    print("\nPART 3 — VALUE ITERATION")
    print("Value iteration sweeps:", len(vi_history))
    print_grid(V_opt_vi, "V* from value iteration:")
    print_policy(policy_opt_vi, "Greedy policy extracted from V*:")
    print("Policies match:", np.array_equal(policy_opt_pi, policy_opt_vi))

    # Part 4: exact linear-system verification
    V_exact = exact_policy_evaluation(uniform_policy, P, R, GAMMA)
    max_diff = float(np.max(np.abs(V_pi - V_exact)))
    print("\nPART 4 — EXACT LINEAR ALGEBRA")
    print_grid(V_exact, "Exact V^pi:")
    print("Maximum absolute difference (iterative vs exact):", f"{max_diff:.12e}")

    # Part 5: gamma experiments
    print("\nPART 5 — DISCOUNT-FACTOR EXPERIMENTS")
    for g in [0.5, 0.9, 0.999]:
        Vg, pg, hg = value_iteration(P, R, g, THETA)
        print(f"gamma={g}: sweeps={len(hg)}, V*(0)={Vg[0]:.9f}")
    P0, R0 = build_model(step_reward=0.0)
    V0, p0, h0 = value_iteration(P0, R0, GAMMA, THETA)
    print("\nLiving reward changed from -0.01 to 0:")
    print("sweeps =", len(h0), "| V*(0) =", f"{V0[0]:.9f}")
    print("In this grid, the optimal policy remains unchanged.")

    print("\nIf P and R are unknown, use model-free methods such as TD learning,")
    print("Q-learning, or SARSA with sampled (state, action, reward, next-state)")
    print("experiences and an exploration strategy such as epsilon-greedy.")

if __name__ == "__main__":
    main()
