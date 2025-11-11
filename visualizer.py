import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def visualize_landing(sim_or_data, policy=None, sim_time=None, dt=None, speedup=1.0, save=False):
    """
    Visualize a 2D rocket landing simulation with telemetry overlay,
    fuel bar, trajectory trace, and grid overlay.

    Supports either:
        - visualize_landing(sim, policy): runs a new simulation
        - visualize_landing(result_dict): plays back a stored trajectory
          from sim.run_episodes() or sim.run_one_episode()
    """
    # Detect mode (simulation vs. playback)
    if isinstance(sim_or_data, dict) and "trajectory" in sim_or_data:
        # --- Playback mode ---
        sim = None
        traj = sim_or_data["trajectory"]
        states = np.array(traj["states"])
        actions = np.array(traj["actions"])
        p = sim_or_data.get("params", None)
        if p is None and "params" in traj:
            p = traj["params"]
        elif p is None:
            raise ValueError("No parameter dictionary found in trajectory data.")
        dt = p.get("dt", 0.02) if dt is None else dt
        frames = len(states)
    else:
        # --- Simulation mode ---
        sim = sim_or_data
        p = sim.params
        if dt is None:
            dt = p["dt"]
        if sim_time is None:
            sim_time = p["max_episode_steps"] * dt

        # ---------------- Pre-simulate physics ----------------
        state, throttle_act, gimbal_act = sim.reset()
        obs = sim.observe(state)
        states, actions = [], []
        for step in range(int(sim_time / dt)):
            action = policy(obs, {"params": p, "step": step})
            state, throttle_act, gimbal_act, _ = sim.step(
                state, throttle_act, gimbal_act, action, dt, 0.0
            )
            states.append(state.copy())
            actions.append(action)
            obs = sim.observe(state)
            if state[1] <= 0:
                break

        states = np.array(states)
        actions = np.array(actions)
        frames = len(states)

    # ---------------- Figure setup ----------------
    fig = plt.figure(figsize=(8, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 0.3])
    ax = fig.add_subplot(gs[0, 0])
    ax_fuel = fig.add_subplot(gs[0, 1])

    ax.set_xlim(-20, 20)
    ax.set_ylim(0, max(50, np.max(states[:, 1]) + 10))
    ax.set_xlabel("Horizontal position (m)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("2D Rocket Landing Simulator")

    # --- GRID OVERLAY ---
    ax.grid(True, which='both', linestyle='--', color='gray', alpha=0.4)

    # Rocket, flame, and trace
    body_len, body_wid = 2.0, 0.4
    rocket_body, = ax.plot([], [], 'k-', lw=3)
    flame, = ax.plot([], [], 'orange', lw=2)
    trace_line, = ax.plot([], [], 'b--', lw=1, alpha=0.5)
    ax.plot([-50, 50], [0, 0], 'g-', lw=2)  # ground

    # Telemetry overlay
    telemetry = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, fontsize=10,
        va="top", ha="left", color="black",
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none")
    )

    # Fuel bar setup
    ax_fuel.set_xlim(0, 1)
    ax_fuel.set_ylim(0, 1)
    ax_fuel.set_xticks([])
    ax_fuel.set_yticks([])
    ax_fuel.set_title("Fuel", fontsize=10)
    ax_fuel.grid(False)
    fuel_bar = ax_fuel.bar(0.5, 1.0, width=0.8, color="blue")[0]

    # ---------------- Frame update ----------------
    def init():
        rocket_body.set_data([], [])
        flame.set_data([], [])
        trace_line.set_data([], [])
        telemetry.set_text("")
        fuel_bar.set_height(1.0)
        return rocket_body, flame, telemetry, fuel_bar, trace_line

    def update(frame):
        x, z, vx, vz, theta, theta_dot, f = states[frame]
        throttle, gimbal = actions[frame]

        # Rocket geometry
        pts = np.array([
            [-body_wid/2, 0],
            [ body_wid/2, 0],
            [ body_wid/2, body_len],
            [-body_wid/2, body_len],
            [-body_wid/2, 0]
        ])
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        pts_world = pts @ R.T + np.array([x, z])
        rocket_body.set_data(pts_world[:, 0], pts_world[:, 1])

        # Flame
        if throttle > 0.05 and f > 0:
            flame_len = 1.0 * throttle
            base = np.array([x, z])
            tip = base - flame_len * np.array([
                np.sin(theta + gimbal),
                np.cos(theta + gimbal)
            ])
            flame.set_data([base[0], tip[0]], [base[1], tip[1]])
        else:
            flame.set_data([], [])

        # Trace
        trace_line.set_data(states[:frame, 0], states[:frame, 1])

        # Telemetry overlay
        telemetry.set_text(
            f"t={frame*dt:5.1f}s\n"
            f"Alt: {z:6.1f} m\n"
            f"Vx:  {vx:6.2f} m/s\n"
            f"Vz:  {vz:6.2f} m/s\n"
            f"Ang: {np.rad2deg(theta):6.2f}°\n"
            f"Thr: {throttle*100:6.1f}%\n"
            f"Gim: {np.rad2deg(gimbal):6.2f}°\n"
            f"Fuel: {f:6.1f} kg"
        )

        # Fuel bar
        fuel_frac = max(f / p["fuel_mass0"], 0.0)
        fuel_bar.set_height(fuel_frac)

        return rocket_body, flame, telemetry, fuel_bar, trace_line

    # ---------------- Animation ----------------
    playback_interval = dt * 1000 / speedup
    ani = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init,
        blit=True, interval=playback_interval, repeat=False
    )

    if save:
        ani.save("rocket_landing.mp4", fps=int(speedup / dt))
    else:
        plt.show()
