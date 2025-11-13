"""
2D Rocket Landing Simulator
"""

import numpy as np
from typing import Callable, Tuple, Dict
import math
import os

# the parameters we use for basically like everything
# the orders of magnitude in here now are pretty basic but it makes some of the math easier
DEFAULT_PARAMS = {
    # Rocket phsyical parameters
    "dry_mass": 100.0,          # kg, rocket without fuel
    "fuel_mass0": 100.0,        # kg, initial fuel
    "I": 400.0,                 # kg*m^2, moment of inertia about COM
    "max_thrust": 15000.0,      # N, full throttle
    "Isp": 300.0,               # s, specific impulse
    "g0": 9.81,                 # m/s^2
    "thrust_arm": 1.0,          # m, lever arm for torque (distance from COM to thrust line)
    
    # Aerodynamic parameters
    "Cd": 1.0,                  # drag coefficient (for body)
    "A_cross": 1.0,             # m^2 cross-sectional area
    "rho": 1.225,               # kg/m^3
    
    # Time lag and actuator limits for gimbal and throttle
    "max_gimbal": np.deg2rad(15.0),  # radians
    "tau_throttle": 0.05,       # s, time constant for throttle actuator (low = faster)
    "tau_gimbal": 0.05,         # s, time constant for gimbal actuator
    
    # Wind parameters (and variation)
    "wind_vel_mean": 2.0,       # m/s horizontal mean wind (positive = pushing rocket +x)
    "wind_gust_sigma": 2.0,     # m/s, std dev of gusts (per step)
    
    # Integration
    "dt": 0.05,                 # s, simulation timestep
    
    # Observation noise
    "obs_pos_sigma": 0.01,      # m
    "obs_vel_sigma": 0.01,      # m/s
    "obs_angle_sigma": 0.001,   # rad
    "obs_angvel_sigma": 0.001,  # rad/s
    "obs_fuel_sigma": 0.1,      # kg
    
    # Reward / termination thresholds
    "vx_tol": 1.0,              # m/s
    "vz_tol": 1.0,              # m/s
    "theta_tol": np.deg2rad(5.0), # rad
    "max_episode_steps": 20000,  # max steps
    
    # Rewards
    "reward_landing_success": 1000.0,
    "reward_crash": -1000.0,
    "reward_per_step": -1.0,    # we need a penalty for taking too long to land -- helps us balance landing quickly with actually landing
    "fuel_penalty_weight": 0.1, # weight * fuel_used (kg)
    "tilt_penalty_weight": 50.0, # penalty scale for tilt at landing
    "vel_penalty_weight": 10.0,  # penalty scale for touchdown velocities
}


# just some fun functions to make the math more visibly appealing later on

def mass_from_fuel(dry_mass: float, fuel: float) -> float:
    # if we take a time step that drops our mass in the negative, we need to drop our fuel mass to 0
    return dry_mass + max(0.0, fuel)


def fuel_flow_from_thrust(thrust: float, Isp: float, g0: float) -> float:
    # we'll be recalculating you whenever we need to toggle our thrust
    # better helps us model our fuel loss
    return thrust / (Isp * g0 + 1e-12)


# the dynamics! this effectively breaks down our current state, control efforts, and wind disturbances and uses them to figure out 

def rocket_dynamics(state: np.ndarray,
                    applied_throttle: float,
                    applied_gimbal: float,
                    params: Dict,
                    wind_vel: float) -> np.ndarray:
    """
    The actual dynamics of our rocket

    state: [x, z, vx, vz, theta, theta_dot, f]
    applied_throttle: throttle value [0,1]
    applied_gimbal: gimbal angle (rad) relative to body axis
    wind_vel: horizontal wind speed (m/s) (positive to +x)
    returns: state_dot numpy array
    """

    # take apart our state input
    x, z, vx, vz, theta, theta_dot, f = state

    # unpack our parameters dictionary
    dry_mass = params["dry_mass"]
    I = params["I"]
    max_thrust = params["max_thrust"]
    thrust_arm = params["thrust_arm"]
    Cd = params["Cd"]
    A = params["A_cross"]
    rho = params["rho"]
    Isp = params["Isp"]
    g0 = params["g0"]

    # calculare our current mass
    mass = mass_from_fuel(dry_mass, f)

    # figure out how much thrust we end up applying based on how much throttle our policy dictates we give
    throttle = np.clip(applied_throttle, 0.0, 1.0)
    T = throttle * max_thrust

    # Fuel flow
    mdot = fuel_flow_from_thrust(T, Isp, g0)  # kg/s
    # if our fuel use dries up, we lose all thrust and mass flow (duh)
    if f <= 0.0:
        T = 0.0
        mdot = 0.0

    # calculating the drag on the rocket from the wind
    u_rel = vx - wind_vel  # first we need to know the relative wind speed
    # then we calculate the drag: Fd = 0.5 * rho * Cd * A * u_rel * |u_rel|
    Fd_x = -0.5 * rho * Cd * A * u_rel * abs(u_rel)

    # we're assuming the wind just blows side to side, so no vertical wind here 
    # the only drag comes from the wind we're pushing out of the way on our way down
    v_rel_z = vz
    Fd_z = -0.5 * rho * Cd * A * v_rel_z * abs(v_rel_z)


    # we need to break down the thrust we're applying into a vertical and horizontal component
    # the actual angle the thrust leaves at is the combination of the rocket's angle and the gimbal's angle
    total_angle = theta + applied_gimbal
    # thrust components (x horizontal, z vertical up)
    thrust_x = T * math.sin(total_angle)
    thrust_z = T * math.cos(total_angle)

    # translational acceleration
    ax = (thrust_x + Fd_x) / mass
    az = (thrust_z + Fd_z) / mass - g0  # gravity downward

    # torque about COM: assume thrust line offset creates moment proportional to T * lever * sin(gimbal)
    # sign: if gimbal positive (clockwise), torque tends to increase theta_dot (clockwise)
    torque = T * thrust_arm * math.sin(applied_gimbal)
    alpha = torque / I  # angular acceleration

    # state derivatives
    x_dot = vx
    z_dot = vz
    vx_dot = ax
    vz_dot = az
    theta_dot_out = theta_dot
    theta_ddot = alpha
    f_dot = -mdot

    return np.array([x_dot, z_dot, vx_dot, vz_dot, theta_dot_out, theta_ddot, f_dot], dtype=float)


# this is a hard coded version of the Runge-Kutta integration technique
# I know NOW that numpy has hard coded integrators but uh...I didn't think about that before this
# so...here it is. it works, so
def rk4_step(state: np.ndarray,
             throttle_act: float,
             gimbal_act: float,
             params: Dict,
             wind_vel: float,
             dt: float) -> np.ndarray:

    f1 = rocket_dynamics(state, throttle_act, gimbal_act, params, wind_vel)
    s2 = state + 0.5 * dt * f1
    f2 = rocket_dynamics(s2, throttle_act, gimbal_act, params, wind_vel)
    s3 = state + 0.5 * dt * f2
    f3 = rocket_dynamics(s3, throttle_act, gimbal_act, params, wind_vel)
    s4 = state + dt * f3
    f4 = rocket_dynamics(s4, throttle_act, gimbal_act, params, wind_vel)
    new_state = state + (dt / 6.0) * (f1 + 2.0 * f2 + 2.0 * f3 + f4)

    # enforce non-negative fuel
    new_state[6] = max(0.0, new_state[6])
    return new_state


# this is where the magic actually happens
# this class basically wraps together everything to run the simulator over and over again to compare performances under different initial conditions
class RocketSimulator:
    def __init__(self, params: Dict = None, seed: int = 0):
        self.params = DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)
        self.rng = np.random.default_rng(seed)

    # this is what we use when switching between episodes after one's concluded
    def reset(self, init_state: Dict = None):
        # if we don't provide some initial state for the simulator, it just randomly generates it based on the parameters below
        p = self.params
        if init_state is None:
            # a default high-altitude simple initial condition
            x0 = self.rng.normal(0.0, 0.5)
            z0 = self.rng.uniform(200.0, 300.0)  # 200-300m up
            vx0 = self.rng.normal(0.0, 1.0)
            vz0 = self.rng.normal(-5.0, 0.0) # I want this thing to be Flyin down the screen to start
            theta0 = self.rng.normal(0.0, np.deg2rad(5.0))
            theta_dot0 = self.rng.normal(0.0, 0.1)
            f0 = p["fuel_mass0"]
        else:
            x0, z0, vx0, vz0, theta0, theta_dot0, f0 = init_state

        state = np.array([x0, z0, vx0, vz0, theta0, theta_dot0, f0], dtype=float)

        # actuator filtered states (simulate actuation lag)
        throttle_act = 0.0
        gimbal_act = 0.0
        return state, throttle_act, gimbal_act

    # this is how we move the simulator forward one "step" (ha, get it) at a time
    def step(self,
             state: np.ndarray,
             throttle_act: float,
             gimbal_act: float,
             action: Tuple[float, float],
             dt: float,
             wind_vel: float) -> Tuple[np.ndarray, float, float, Dict]:
        """
        Apply action commands through actuator lag, step dynamics one dt.
        action: (throttle_cmd, gimbal_cmd)
        Returns new_state, new_throttle_act, new_gimbal_act, info dict.
        """
        p = self.params
        throttle_cmd, gimbal_cmd = action
        # clamp commands
        throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))
        gimbal_cmd = float(np.clip(gimbal_cmd, -p["max_gimbal"], p["max_gimbal"]))

        # first-order actuator dynamics: x_dot = (cmd - x)/tau
        tau_t = p["tau_throttle"]
        tau_g = p["tau_gimbal"]
        if tau_t <= 0:
            new_throttle_act = throttle_cmd
        else:
            new_throttle_act = throttle_act + (dt / tau_t) * (throttle_cmd - throttle_act)
        if tau_g <= 0:
            new_gimbal_act = gimbal_cmd
        else:
            new_gimbal_act = gimbal_act + (dt / tau_g) * (gimbal_cmd - gimbal_act)

        # Integrate dynamics using RK4 with the filtered actuator values held constant during dt
        new_state = rk4_step(state, new_throttle_act, new_gimbal_act, p, wind_vel, dt)

        # book-keeping schtuff
        info = {
            "throttle_act": new_throttle_act,
            "gimbal_act": new_gimbal_act,
            "wind_vel": wind_vel,
            "thrust": new_throttle_act * p["max_thrust"] if new_state[6] > 0 else 0.0,
            "mdot": fuel_flow_from_thrust(new_throttle_act * p["max_thrust"], p["Isp"], p["g0"]) if new_state[6] > 0 else 0.0
        }
        return new_state, new_throttle_act, new_gimbal_act, info

    # what makes this a POMDP and not just an MDP is that our observations are imperfect
    # soooooo, we inject a little noise into our sensors in the form of a zero mean Gaussian
    def observe(self, true_state: np.ndarray) -> np.ndarray:
        p = self.params
        obs = true_state.copy()
        obs[0] += self.rng.normal(0.0, p["obs_pos_sigma"])
        obs[1] += self.rng.normal(0.0, p["obs_pos_sigma"])
        obs[2] += self.rng.normal(0.0, p["obs_vel_sigma"])
        obs[3] += self.rng.normal(0.0, p["obs_vel_sigma"])
        obs[4] += self.rng.normal(0.0, p["obs_angle_sigma"])
        obs[5] += self.rng.normal(0.0, p["obs_angvel_sigma"])
        obs[6] += self.rng.normal(0.0, p["obs_fuel_sigma"])
        obs[6] = max(0.0, obs[6])
        return obs

    def reward_and_done(self, state: np.ndarray, step_count: int) -> Tuple[float, bool, Dict]:
        # this function does a bunch of checks to see if we've "finished" our run (i.e. we run out of fuel and fall to the ground or if we land successfully)
        # it also generates a reward with every time step -- those reward values are found up above in our parameters section
        p = self.params
        x, z, vx, vz, theta, theta_dot, f = state

        # default per-step penalty (encourage faster landings and less time wasted)
        r = p["reward_per_step"]
        
        # we're running into problems where the rocket keeps tilting and dropping too fast
        # so, hopefully this stops that (please please please)
        # Penalize horizontal drift and tilt
        r -= 0.05 * abs(x)                 # stay near centerline
        r -= 5.0  * abs(theta)             # stay upright

        # Penalize vertical and horizontal velocity (encourages slowing down)
        r -= 0.2  * abs(vz)
        r -= 0.05 * abs(vx)

        # Small positive reward for staying higher (avoids immediate crash)
        r += 0.01 * z

        done = False
        info = {"success": False, "crash": False}
        alt_tol = 0.5  # meters, altitude tolerance for considering "landed"

        if z <= alt_tol:
            # touchdown checks
            tilt_pen = p["tilt_penalty_weight"] * (abs(theta) / (p["theta_tol"] + 1e-8))
            vel_pen = p["vel_penalty_weight"] * (abs(vz) / (p["vz_tol"] + 1e-8) + abs(vx) / (p["vx_tol"] + 1e-8))
            r -= (tilt_pen + vel_pen)

            # success criteria
            success = (
                abs(vz) <= p["vz_tol"] and
                abs(vx) <= p["vx_tol"] and
                abs(theta) <= p["theta_tol"] and
                z <= alt_tol
            )

            if success:
                r += p["reward_landing_success"]
                info["success"] = True
            else:
                r += p["reward_crash"]
                info["crash"] = True
            done = True

        elif step_count >= p["max_episode_steps"]:
            # Treat timeout as crash
            done = True
            r += p["reward_crash"] * 0.1
            info["crash"] = True

        # fuel penalty is applied externally per-step depending on fuel used; the environment returns fuel used in info too.
        return r, done, info

    def run_one_episode(self,
                        policy: Callable[[np.ndarray, Dict], Tuple[float, float]],
                        init_state: Dict = None,
                        record_trajectory: bool = True,
                        deterministic: bool = True) -> Dict:
        """
        Runs one episode with given policy function.
        policy(obs, info) -> (throttle_cmd, gimbal_cmd)
        Returns dictionary with trajectory data and final info.
        """
        
        # this is where we cycle through the length of one run
        # we have some stochastic initial state and some policy and we see how the system performs
        
        p = self.params
        dt = p["dt"]
        state, throttle_act, gimbal_act = self.reset(init_state)
        obs = self.observe(state)
        step = 0
        traj = {
            "states": [],
            "obs": [],
            "actions": [],
            "infos": [],
            "rewards": []
        }
        total_reward = 0.0
        total_fuel_used = 0.0

        # initial wind
        wind_vel = self.rng.normal(p["wind_vel_mean"], p["wind_gust_sigma"])

        while True:
            # policy chooses based on observation (POMDP)
            # NOTE: the action is, in order, the throttle command and gimbal command. This is what should be returned at the end of our guidance policy.
            
            action = policy(obs, {"params": p, "step": step})
            # ensure action shape
            throttle_cmd, gimbal_cmd = action

            # update wind each step (simple random walk/gust)
            wind_vel = self.rng.normal(p["wind_vel_mean"], p["wind_gust_sigma"])

            new_state, throttle_act, gimbal_act, info = self.step(state, throttle_act, gimbal_act, (throttle_cmd, gimbal_cmd), dt, wind_vel)

            # compute fuel used this step
            fuel_used = max(0.0, state[6] - new_state[6])
            total_fuel_used += fuel_used

            # using our reward_and_done function, we tack on the newly gained reward to our current reward
            r, done, terminal_info = self.reward_and_done(new_state, step)
            # add fuel penalty
            r -= p["fuel_penalty_weight"] * fuel_used

            total_reward += r

            if record_trajectory:
                traj["states"].append(state.copy())
                traj["obs"].append(obs.copy())
                traj["actions"].append(np.array([throttle_cmd, gimbal_cmd], dtype=float))
                traj["infos"].append(info)
                traj["rewards"].append(r)

            # step forward
            state = new_state
            obs = self.observe(state)
            step += 1

            if done:
                final_info = {"terminal": terminal_info, "total_reward": total_reward, "steps": step, "fuel_used": total_fuel_used}
                
                print(f"Episode finished after {step} steps. Total reward = {total_reward:.2f}")
                
                if record_trajectory:
                    traj["states"].append(state.copy())
                    traj["obs"].append(obs.copy())
                return {"trajectory": traj, "final_info": final_info}

    def run_episodes(self, policy: Callable, n_episodes: int, save_to: str = None):
        # this runs a bunch of single episodes and then saves the output to a npz file
        results = []
        landing_tilts = []
        landing_vxs = []
        landing_vzs = []

        for i in range(n_episodes):
            res = self.run_one_episode(policy)
            results.append(res)

            final_state = res["trajectory"]["states"][-1]
            x, z, vx, vz, theta, theta_dot, f = final_state #access velocity and tilt angle

            landing_tilts.append(theta)
            landing_vxs.append(vx)
            landing_vzs.append(vz)

            print(
                f"Episode {i+1}: total reward = {res['final_info']['total_reward']:.2f}, "
                f"success = {res['final_info']['terminal']['success']}, "
                f"tilt = {np.rad2deg(theta):.2f} deg, "
                f"vx = {vx:.2f} m/s, vz = {vz:.2f} m/s"
            )

            if (i+1) % 50 == 0:
                print(f"Completed {i+1}/{n_episodes}")

        # --- Aggregate Statistics ---
        total_rewards = [r["final_info"]["total_reward"] for r in results]
        successes = [r["final_info"]["terminal"]["success"] for r in results]
        crashes = [r["final_info"]["terminal"]["crash"] for r in results]
        fuel_used = [r["final_info"]["fuel_used"] for r in results]

        avg_reward = np.mean(total_rewards)
        median_reward = np.median(total_rewards)
        success_rate = np.mean(successes) * 100
        crash_rate = np.mean(crashes) * 100
        avg_fuel = np.mean(fuel_used)

        mean_tilt_deg = np.rad2deg(np.mean(landing_tilts))
        mean_vx = np.mean(landing_vxs)
        mean_vz = np.mean(landing_vzs)

        print("\n=== Simulation Summary ===")
        print(f"Episodes run:       {len(results)}")
        print(f"Average reward:     {avg_reward:.2f}")
        print(f"Median reward:      {median_reward:.2f}")
        print(f"Success rate:       {success_rate:.1f}%")
        print(f"Crash rate:         {crash_rate:.1f}%")
        print(f"Average fuel used:  {avg_fuel:.2f} kg")

        #add in final velocity and tilt
        print(f"Mean landing tilt angle:    {mean_tilt_deg:.2f} deg")
        print(f"Mean landing velocity (x):      {mean_vx:.2f} m/s")
        print(f"Mean landing velocity (z):      {mean_vz:.2f} m/s")
        print("==========================\n")

        if save_to:
            np.savez_compressed(save_to, episodes=results)
            print(f"Saved {len(results)} episodes to {save_to}")

        return results
