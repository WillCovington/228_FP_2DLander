import csv
import numpy as np
import matplotlib.pyplot as plt

'''This file contains the code to plot the true state vs. the estimated state found from
our Kalman Filter in kf_wrapper.py. We compare the true state vs. the estimated state for 
x, vx, z, vz, theta, theta_dot, and fuel. We also plot the KF estimation errors for each of
these parameters.'''

CSV_FILE = "data/KF_state_data.csv"

def load_kf_data(filename: str):
    """
    Load KF state CSV with columns:
    step, true_*, est_*, err_*.
    @return: steps: (N,), true_states: (N, 7), est_states: (N, 7), err_states: (N, 7)
    """
    with open(filename, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        # figure out column indices by name
        names = ["x", "z", "vx", "vz", "theta", "theta_dot", "fuel"]
        idx_step = header.index("step")
        idx_true = [header.index(f"true_{n}") for n in names]
        idx_est  = [header.index(f"est_{n}") for n in names]
        idx_err  = [header.index(f"err_{n}") for n in names]

        steps = []
        true_list = []
        est_list = []
        err_list = []

        for row in reader:
            if not row:
                continue
            steps.append(int(row[idx_step]))
            true_list.append([float(row[i]) for i in idx_true])
            est_list.append([float(row[i]) for i in idx_est])
            err_list.append([float(row[i]) for i in idx_err])

    steps = np.array(steps)
    true_states = np.array(true_list)   # (N,7)
    est_states  = np.array(est_list)    # (N,7)
    err_states  = np.array(err_list)    # (N,7)
    return steps, true_states, est_states, err_states, names

def plot_states(steps, true_states, est_states, names):
    """
    Plot true vs estimated states over time.
    """
    n_states = true_states.shape[1]
    n_rows = 4
    n_cols = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for i in range(n_states):
        ax = axes[i]
        ax.plot(steps, true_states[:, i], label="True", linewidth=1.5)
        ax.plot(steps, est_states[:, i], label="Estimated", linestyle="--", linewidth=1.2)
        ax.set_title(names[i])
        ax.grid(True, alpha=0.3)

        if i % n_cols == 0:
            ax.set_ylabel("Value")

    # hide any unused subplot
    for j in range(n_states, len(axes)):
        axes[j].axis("off")

    axes[-2].set_xlabel("Step")
    axes[-1].set_xlabel("Step")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("True vs Estimated States (Kalman Filter)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.98, 0.95])
    fig.savefig("kf_states.png", dpi=200)
    plt.show()

def plot_errors(steps, err_states, names):
    """
    Plot estimation error for each state over time.
    """
    n_states = err_states.shape[1]
    n_rows = 4
    n_cols = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for i in range(n_states):
        ax = axes[i]
        ax.plot(steps, err_states[:, i], linewidth=1.2)
        ax.set_title(f"Error: {names[i]} (est - true)")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.3)

        if i % n_cols == 0:
            ax.set_ylabel("Error")

    # hide any unused subplot
    for j in range(n_states, len(axes)):
        axes[j].axis("off")

    axes[-2].set_xlabel("Step")
    axes[-1].set_xlabel("Step")

    fig.suptitle("Kalman Filter Estimation Errors", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("kf_errors.png", dpi=200)
    plt.show()

def main():
    '''Execute analysis and plotting code!'''
    steps, true_states, est_states, err_states, names = load_kf_data(CSV_FILE)
    print(f"Loaded {len(steps)} samples from {CSV_FILE}")
    plot_states(steps, true_states, est_states, names)
    plot_errors(steps, err_states, names)

if __name__ == "__main__":
    main()