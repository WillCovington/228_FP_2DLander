import re
import matplotlib.pyplot as plt
import numpy as np

# Path to your log file
LOG_PATH = "trials/baseline_policy2.txt"

def parse_log(path):
    episode_ids = []
    episode_rewards = []
    episode_success = []

    # per-episode landing metrics
    episode_tilts_deg = []
    episode_vx = []
    episode_vz = []

    avg_reward = None
    avg_fuel = None
    crash_rate = None
    runtime = None

    last_episode_index = None  # index of the most recently seen episode

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            # --- Episode line (reward + success) ---
            m_ep = re.match(
                r"Episode\s+(\d+):\s*total reward\s*=\s*([\-0-9\.]+),\s*success\s*=\s*(True|False)",
                line
            )
            if m_ep:
                ep_id = int(m_ep.group(1))
                rew = float(m_ep.group(2))
                success = (m_ep.group(3) == "True")

                episode_ids.append(ep_id)
                episode_rewards.append(rew)
                episode_success.append(success)

                # placeholders for this episode's tilt/vels
                episode_tilts_deg.append(None)
                episode_vx.append(None)
                episode_vz.append(None)

                last_episode_index = len(episode_ids) - 1

                # fall through to also parse tilt/vx/vz if they’re on the same line
                # (no continue here)

            # --- Look for tilt / vx / vz on ANY line (including Episode line) ---
            # Attach them to the last seen episode, if any.
            if last_episode_index is not None:
                mt = re.search(r"tilt\s*=\s*([\-0-9\.]+)\s*deg", line)
                mvx = re.search(r"vx\s*=\s*([\-0-9\.]+)\s*m/s", line)
                mvz = re.search(r"vz\s*=\s*([\-0-9\.]+)\s*m/s", line)

                if mt:
                    episode_tilts_deg[last_episode_index] = float(mt.group(1))
                if mvx:
                    episode_vx[last_episode_index] = float(mvx.group(1))
                if mvz:
                    episode_vz[last_episode_index] = float(mvz.group(1))

            # --- Summary lines ---
            m_avg_r = re.match(r"Average reward:\s*([\-0-9\.]+)", line)
            if m_avg_r:
                avg_reward = float(m_avg_r.group(1))
                continue

            m_fuel = re.match(r"Average fuel used:\s*([0-9\.]+)\s*kg", line)
            if m_fuel:
                avg_fuel = float(m_fuel.group(1))
                continue

            m_crash = re.match(r"Crash rate:\s*([0-9\.]+)%", line)
            if m_crash:
                crash_rate = float(m_crash.group(1))
                continue

            m_runtime = re.match(r"Total runtime:\s*([0-9\.]+)\s*seconds", line)
            if m_runtime:
                runtime = float(m_runtime.group(1))
                continue

    # Fallback averages
    if avg_reward is None and episode_rewards:
        avg_reward = np.mean(episode_rewards)

    if crash_rate is None and episode_success:
        n_episodes = len(episode_success)
        n_crashes = sum(1 for s in episode_success if not s)
        crash_rate = 100.0 * n_crashes / n_episodes if n_episodes > 0 else 0.0

    # Reward variance
    reward_variance = np.var(episode_rewards) if episode_rewards else None

    return {
        "episode_ids": episode_ids,
        "episode_rewards": episode_rewards,
        "episode_success": episode_success,
        "episode_tilts_deg": episode_tilts_deg,
        "episode_vx": episode_vx,
        "episode_vz": episode_vz,
        "avg_reward": avg_reward,
        "avg_fuel": avg_fuel,
        "crash_rate": crash_rate,
        "runtime": runtime,
        "reward_variance": reward_variance
    }


def make_plots(stats):
    episode_ids = stats["episode_ids"]
    episode_rewards = stats["episode_rewards"]
    avg_reward = stats["avg_reward"]
    avg_fuel = stats["avg_fuel"]
    crash_rate = stats["crash_rate"]
    runtime = stats["runtime"]
    reward_variance = stats["reward_variance"]

    episode_tilts_deg = stats["episode_tilts_deg"]
    episode_vx = stats["episode_vx"]
    episode_vz = stats["episode_vz"]

    # Helper: label a horizontal line near (x_ref, y_val)
    def label_line(ax, x_ref, y_val, text):
        if y_val == 0:
            y_text = 0.02
            va = "bottom"
        else:
            offset = 0.02 * abs(y_val) or 0.05
            if y_val > 0:
                y_text = y_val + offset
                va = "bottom"
            else:
                y_text = y_val - offset
                va = "top"
        ax.text(
            x_ref, y_text, text,
            color="green", fontsize=10, fontweight="bold", va=va
        )

    # ============================================================
    # Figure 1: Rewards + aggregate metrics
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"KF Rollout Policy Summary (Runtime: {runtime:.2f} s)"
        if runtime is not None else "KF Rollout Policy Summary",
        fontsize=14
    )

    # 1) Reward per episode (thin-ish bars + line)
    ax0 = axes[0, 0]
    if episode_ids and episode_rewards:
        bar_width = 0.3
        ax0.bar(
            episode_ids,
            episode_rewards,
            width=bar_width,
            alpha=0.6,
            edgecolor="black",
            label="Reward"
        )
        ax0.plot(
            episode_ids,
            episode_rewards,
            marker="o",
            linewidth=1.5,
            label="Trend"
        )
        ax0.set_xlabel("Episode")
        ax0.set_ylabel("Total Reward")
        ax0.set_title("Reward per Episode")
        ax0.legend()
    else:
        ax0.text(0.5, 0.5, "No episode data", ha="center", va="center")
        ax0.set_axis_off()

    # 2) Average reward
    ax1 = axes[0, 1]
    if avg_reward is not None:
        ax1.bar([0], [avg_reward], width=0.03, edgecolor="black")
        ax1.axhline(avg_reward, color="green", linestyle="-", linewidth=1.5)
        label_line(ax1, x_ref=0.05, y_val=avg_reward, text=f"{avg_reward:.2f}")
        ax1.set_xticks([0])
        ax1.set_xticklabels(["Average"])
        ax1.set_xlim(-0.5, 0.5)
        ax1.set_ylabel("Reward")
        ax1.set_title("Average Reward")
    else:
        ax1.text(0.5, 0.5, "No average reward", ha="center", va="center")
        ax1.set_axis_off()

    # 3) Reward variance
    ax2 = axes[0, 2]
    if reward_variance is not None:
        ax2.bar([0], [reward_variance], width=0.03, edgecolor="black")
        ax2.axhline(reward_variance, color="green", linestyle="-", linewidth=1.5)
        label_line(ax2, x_ref=0.05, y_val=reward_variance, text=f"{reward_variance:.2f}")
        ax2.set_xticks([0])
        ax2.set_xticklabels(["Variance"])
        ax2.set_xlim(-0.5, 0.5)
        ax2.set_ylabel("Variance")
        ax2.set_title("Reward Variance")
    else:
        ax2.text(0.5, 0.5, "No variance data", ha="center", va="center")
        ax2.set_axis_off()

    # 4) Average fuel
    ax3 = axes[1, 0]
    if avg_fuel is not None:
        ax3.bar([0], [avg_fuel], width=0.03, edgecolor="black")
        ax3.axhline(avg_fuel, color="green", linestyle="-", linewidth=1.5)
        label_line(ax3, x_ref=0.05, y_val=avg_fuel, text=f"{avg_fuel:.2f}")
        ax3.set_xticks([0])
        ax3.set_xticklabels(["Fuel"])
        ax3.set_xlim(-0.5, 0.5)
        ax3.set_ylabel("Fuel (kg)")
        ax3.set_title("Average Fuel Used")
    else:
        ax3.text(0.5, 0.5, "No fuel data", ha="center", va="center")
        ax3.set_axis_off()

    # 5) Crash rate
    ax4 = axes[1, 1]
    if crash_rate is not None:
        ax4.bar([0], [crash_rate], width=0.03, edgecolor="black")
        ax4.set_xticks([0])
        ax4.set_xticklabels(["Crash"])
        ax4.set_xlim(-0.5, 0.5)
        ax4.set_ylabel("Crash Rate (%)")
        ax4.set_title("Crash Rate")

        ax4.axhline(0, color="gray", linestyle="--", linewidth=1)

        if crash_rate == 0:
            ax4.text(
                0, 0.02, "0%",
                ha="center", va="bottom",
                fontsize=10, color="green", fontweight="bold"
            )
            ax4.set_ylim(0, 10)
        else:
            label_line(ax4, x_ref=0.05, y_val=crash_rate, text=f"{crash_rate:.2f}%")
            ax4.set_ylim(0, max(10, crash_rate * 1.5))
    else:
        ax4.text(0.5, 0.5, "No crash data", ha="center", va="center")
        ax4.set_axis_off()

    axes[1, 2].set_axis_off()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # ============================================================
    # Figure 2: Per-episode tilt and velocities
    # ============================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle("Landing State per Episode", fontsize=14)

    def filtered_xy(values):
        pairs = [(ep, v) for ep, v in zip(episode_ids, values) if v is not None]
        if not pairs:
            return [], []
        xs, ys = zip(*pairs)
        return list(xs), list(ys)

    def plot_metric(ax, values, title, ylabel):
        xs, ys = filtered_xy(values)
        if xs and ys:
            bar_width = 0.3
            ax.bar(xs, ys, width=bar_width, alpha=0.6,
                   edgecolor="black", label=ylabel)
            ax.plot(xs, ys, marker="o", linewidth=1.5, label="Trend")
            ax.set_xlabel("Episode")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
        else:
            ax.text(0.5, 0.5, f"No {ylabel} data", ha="center", va="center")
            ax.set_axis_off()

    plot_metric(axes2[0], episode_tilts_deg, "Landing Tilt per Episode", "Tilt (deg)")
    plot_metric(axes2[1], episode_vx, "Landing vx per Episode", "vx (m/s)")
    plot_metric(axes2[2], episode_vz, "Landing vz per Episode", "vz (m/s)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    stats = parse_log(LOG_PATH)
    print("Parsed stats:", stats)
    print(f"Reward Variance: {stats['reward_variance']}")
    make_plots(stats)
