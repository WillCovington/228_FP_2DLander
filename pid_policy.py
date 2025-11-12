import numpy as np
from typing import Tuple, Dict, Sequence
import math
from dataclasses import dataclass
from rocket_simulator import rk4_step  #uses integrator  :contentReference[oaicite:3]{index=3}

def simple_guidance_policy(obs: np.ndarray, info: Dict) -> Tuple[float, float]:
    """
    A naive guidance policy:
     - target vertical velocity and horizontal position biases proportional to altitude
     - tries to keep the rocket upright with a PD controller on angle
     - returns throttle_cmd in [0,1] and small gimbal command limited in magnitude
    """
    params = info["params"]
    # unpack obs (noisy)
    x, z, vx, vz, theta, theta_dot, f = obs
    # simple vertical guidance: aim for soft descent: vz_target = -min(5, z*0.02)
    vz_target = -min(5.0, z * 0.02)
    # PD compute thrust necessary (very approximate)
    # desired vertical accel a_des = (vz_target - vz) * k + gravity compensation
    k_v = 1.0
    a_des = (vz_target - vz) * k_v + params["g0"]
    mass_est = params["dry_mass"] + max(1e-3, f)
    thrust_required = mass_est * a_des
    throttle_cmd = thrust_required / params["max_thrust"]
    throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))

    # simple horizontal correction via gimbal: want to reduce vx and x error
    kx = 0.1
    kv = 0.2
    x_target = 0.0
    x_err = x - x_target
    # desired horizontal accel
    ax_des = -(kx * x_err + kv * vx)
    # convert desired horizontal acceleration to gimbal roughly:
    # ax = (T/m)*sin(theta + gimbal) ~ (T/m)*(theta + gimbal) for small angles
    T = throttle_cmd * params["max_thrust"]
    if T < 1.0:
        gimbal_cmd = 0.0
    else:
        gimbal_cmd = math.asin(np.clip((ax_des * mass_est) / (T + 1e-6), -0.99, 0.99)) - theta
    # keep gimbal small
    gimbal_cmd = float(np.clip(gimbal_cmd, -params["max_gimbal"], params["max_gimbal"]))

    # angle stabilization (small correction)
    k_ang = 0.5
    k_ang_d = 0.05
    angle_corr = -(k_ang * theta + k_ang_d * theta_dot)
    gimbal_cmd += np.clip(angle_corr, -params["max_gimbal"], params["max_gimbal"])

    return throttle_cmd, gimbal_cmd

def refined_guidance_policy(obs: np.ndarray, info: Dict) -> Tuple[float, float]:
    p = info["params"]
    x, z, vx, vz, theta, theta_dot, f = obs

    # --- adaptive vertical velocity target ---
    if z > 100:
        vz_target = -min(25.0, 0.2 * z)
    elif z > 50:
        vz_target = -10.0
    else:
        vz_target = -min(3.0, 0.05 * z)

    # --- throttle control ---
    k_v = 1.0
    a_des = (vz_target - vz) * k_v + p["g0"]
    mass_est = p["dry_mass"] + max(1e-3, f)
    thrust_required = mass_est * a_des
    throttle_cmd = thrust_required / p["max_thrust"]

    # flare boost
    if z < 50.0 and vz < -5.0:
        throttle_cmd += 0.2
    throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))

    # --- gimbal control ---
    kx, kv = 0.1, 0.2
    ax_des = -(kx * x + kv * vx)
    T = throttle_cmd * p["max_thrust"]
    if T < 1.0:
        gimbal_cmd = 0.0
    else:
        gimbal_cmd = math.asin(np.clip((ax_des * mass_est) / (T + 1e-6), -0.99, 0.99)) - theta

    # --- angle stabilization ---
    k_ang, k_ang_d = 0.5, 0.05
    angle_corr = -(k_ang * theta + k_ang_d * theta_dot)
    gimbal_cmd += np.clip(angle_corr, -p["max_gimbal"], p["max_gimbal"])
    gimbal_cmd = float(np.clip(gimbal_cmd, -p["max_gimbal"], p["max_gimbal"]))

    return throttle_cmd, gimbal_cmd

@dataclass
class RolloutCfg:
    horizon: int = 5
    discount: float = 0.99
    nsamples: int = 20
    throttle_scale: Sequence[float] = (0.85, 1.0, 1.15)
    gimbal_offset_deg: Sequence[float] = (-2.0, 0.0, 2.0)
    seed: int = 123

def rollout_guidance_policy(obs: np.ndarray, info: Dict) -> Tuple[float, float]:
    """
    Short-horizon forward search (rollout) around the baseline PD policy.
    Signature matches simple_guidance_policy: (obs, info) -> (throttle, gimbal).
    """
    params: Dict = info["params"]
    cfg: RolloutCfg = info.get("rollout_cfg", RolloutCfg())
    rng = np.random.default_rng(cfg.seed)

    #baseline (PD) action (simple_guidance_policy is a simple, hand-tuned controller
    #that serves as a baseline comparison)
    base_u, base_phi = simple_guidance_policy(obs, info)

    #small candidate set around baseline
    actions = []
    for s in cfg.throttle_scale:
        u = float(np.clip(base_u * s, 0.0, 1.0))
        for deg in cfg.gimbal_offset_deg:
            phi = base_phi + math.radians(deg)
            phi = float(np.clip(phi, -params["max_gimbal"], params["max_gimbal"]))
            actions.append((u, phi))

    #evaluate by Monte-Carlo rollouts
    vals = np.zeros(len(actions), dtype=float)
    for i, a0 in enumerate(actions):
        ret = 0.0
        for _ in range(cfg.nsamples):
            ret += _rollout_once(obs, a0, params, cfg, rng)
        vals[i] = ret / cfg.nsamples

    #finally, pick best!
    best = int(np.argmax(vals))
    return actions[best]

#helper functions defined as follows:

def _rollout_once(root_obs: np.ndarray, a0: Tuple[float, float],
                  params: Dict, cfg: RolloutCfg, rng: np.random.Generator) -> float:
    s = np.array(root_obs, dtype=float, copy=True)
    total = 0.0
    gamma = 1.0

    #step 1 (candidate action)
    s = _rk4_with_wind(s, a0, params, rng)
    total += gamma * _shape_reward(s, a0, params)
    gamma *= cfg.discount

    #remaining steps (follow baseline PD)
    for _ in range(1, cfg.horizon):
        if _terminal(s, params):
            break
        a = simple_guidance_policy(s, {"params": params})
        s = _rk4_with_wind(s, a, params, rng)
        total += gamma * _shape_reward(s, a, params)
        gamma *= cfg.discount

    return total

def _rk4_with_wind(state: np.ndarray, action: Tuple[float, float],
                   params: Dict, rng: np.random.Generator) -> np.ndarray:
    dt = params["dt"]
    wind = rng.normal(params["wind_vel_mean"], params["wind_gust_sigma"])
    u, phi = action
    #NOTE: rollout assumes instantaneous actuators for speed (same across candidates)
    ns = rk4_step(state, u, phi, params, wind, dt)
    ns[6] = max(0.0, ns[6])
    return ns

def _terminal(state: np.ndarray, params: Dict) -> bool:
    x, z, vx, vz, theta, theta_dot, f = state
    if z <= 0.0:
        return True
    if abs(theta) > 3*params["max_gimbal"] or abs(x) > 1e3:
        return True
    return False

def _shape_reward(state: np.ndarray, action: Tuple[float, float], params: Dict) -> float:
    x, z, vx, vz, theta, theta_dot, f = state
    u, phi = action
    return (
        -0.05 * abs(u)
        - 0.001 * (x**2 + z**2)
        - 0.005 * (vx**2 + vz**2)
        - 0.002 * (theta**2 + theta_dot**2)
    )

import numpy as np
from typing import Callable, Tuple, Dict
import math

def simple_guidance_policy(obs: np.ndarray, info: Dict) -> Tuple[float, float]:
    """
    A naive guidance policy:
     - target vertical velocity and horizontal position biases proportional to altitude
     - tries to keep the rocket upright with a PD controller on angle
     - returns throttle_cmd in [0,1] and small gimbal command limited in magnitude
    """
    params = info["params"]
    # unpack obs (noisy)
    x, z, vx, vz, theta, theta_dot, f = obs
    # simple vertical guidance: aim for soft descent: vz_target = -min(5, z*0.02)
    vz_target = -min(5.0, z * 0.02)
    # PD compute thrust necessary (very approximate)
    # desired vertical accel a_des = (vz_target - vz) * k + gravity compensation
    k_v = 1.0
    a_des = (vz_target - vz) * k_v + params["g0"]
    mass_est = params["dry_mass"] + max(1e-3, f)
    thrust_required = mass_est * a_des
    throttle_cmd = thrust_required / params["max_thrust"]
    throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))

    # simple horizontal correction via gimbal: want to reduce vx and x error
    kx = 0.1
    kv = 0.2
    x_target = 0.0
    x_err = x - x_target
    # desired horizontal accel
    ax_des = -(kx * x_err + kv * vx)
    # convert desired horizontal acceleration to gimbal roughly:
    # ax = (T/m)*sin(theta + gimbal) ~ (T/m)*(theta + gimbal) for small angles
    T = throttle_cmd * params["max_thrust"]
    if T < 1.0:
        gimbal_cmd = 0.0
    else:
        gimbal_cmd = math.asin(np.clip((ax_des * mass_est) / (T + 1e-6), -0.99, 0.99)) - theta
    # keep gimbal small
    gimbal_cmd = float(np.clip(gimbal_cmd, -params["max_gimbal"], params["max_gimbal"]))

    # angle stabilization (small correction)
    k_ang = 0.5
    k_ang_d = 0.05
    angle_corr = -(k_ang * theta + k_ang_d * theta_dot)
    gimbal_cmd += np.clip(angle_corr, -params["max_gimbal"], params["max_gimbal"])

    return throttle_cmd, gimbal_cmd

def refined_guidance_policy(obs: np.ndarray, info: Dict) -> Tuple[float, float]:
    p = info["params"]
    x, z, vx, vz, theta, theta_dot, f = obs

    # --- adaptive vertical velocity target ---
    if z > 100:
        vz_target = -min(25.0, 0.2 * z)
    elif z > 50:
        vz_target = -10.0
    else:
        vz_target = -min(3.0, 0.05 * z)

    # --- throttle control ---
    k_v = 1.0
    a_des = (vz_target - vz) * k_v + p["g0"]
    mass_est = p["dry_mass"] + max(1e-3, f)
    thrust_required = mass_est * a_des
    throttle_cmd = thrust_required / p["max_thrust"]

    # flare boost
    if z < 50.0 and vz < -5.0:
        throttle_cmd += 0.2
    throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))

    # --- gimbal control ---
    kx, kv = 0.1, 0.2
    ax_des = -(kx * x + kv * vx)
    T = throttle_cmd * p["max_thrust"]
    if T < 1.0:
        gimbal_cmd = 0.0
    else:
        gimbal_cmd = math.asin(np.clip((ax_des * mass_est) / (T + 1e-6), -0.99, 0.99)) - theta

    # --- angle stabilization ---
    k_ang, k_ang_d = 0.5, 0.05
    angle_corr = -(k_ang * theta + k_ang_d * theta_dot)
    gimbal_cmd += np.clip(angle_corr, -p["max_gimbal"], p["max_gimbal"])
    gimbal_cmd = float(np.clip(gimbal_cmd, -p["max_gimbal"], p["max_gimbal"]))

    return throttle_cmd, gimbal_cmd

@dataclass
class RolloutCfg:
    horizon: int = 5
    discount: float = 0.99
    nsamples: int = 20
    throttle_scale: Sequence[float] = (0.85, 1.0, 1.15)
    gimbal_offset_deg: Sequence[float] = (-2.0, 0.0, 2.0)
    seed: int = 123

def rollout_guidance_policy(obs: np.ndarray, info: Dict) -> Tuple[float, float]:
    """
    Short-horizon forward search (rollout) around the baseline PD policy.
    Signature matches simple_guidance_policy: (obs, info) -> (throttle, gimbal).
    """
    params: Dict = info["params"]
    cfg: RolloutCfg = info.get("rollout_cfg", RolloutCfg())
    rng = np.random.default_rng(cfg.seed)

    #baseline (PD) action (simple_guidance_policy is a simple, hand-tuned controller
    #that serves as a baseline comparison)
    base_u, base_phi = simple_guidance_policy(obs, info)

    #small candidate set around baseline
    actions = []
    for s in cfg.throttle_scale:
        u = float(np.clip(base_u * s, 0.0, 1.0))
        for deg in cfg.gimbal_offset_deg:
            phi = base_phi + math.radians(deg)
            phi = float(np.clip(phi, -params["max_gimbal"], params["max_gimbal"]))
            actions.append((u, phi))

    #evaluate by Monte-Carlo rollouts
    vals = np.zeros(len(actions), dtype=float)
    for i, a0 in enumerate(actions):
        ret = 0.0
        for _ in range(cfg.nsamples):
            ret += _rollout_once(obs, a0, params, cfg, rng)
        vals[i] = ret / cfg.nsamples

    #finally, pick best!
    best = int(np.argmax(vals))
    return actions[best]

#helper functions defined as follows:

def _rollout_once(root_obs: np.ndarray, a0: Tuple[float, float],
                  params: Dict, cfg: RolloutCfg, rng: np.random.Generator) -> float:
    s = np.array(root_obs, dtype=float, copy=True)
    total = 0.0
    gamma = 1.0

    #step 1 (candidate action)
    s = _rk4_with_wind(s, a0, params, rng)
    total += gamma * _shape_reward(s, a0, params)
    gamma *= cfg.discount

    #remaining steps (follow baseline PD)
    for _ in range(1, cfg.horizon):
        if _terminal(s, params):
            break
        a = simple_guidance_policy(s, {"params": params})
        s = _rk4_with_wind(s, a, params, rng)
        total += gamma * _shape_reward(s, a, params)
        gamma *= cfg.discount

    return total

def _rk4_with_wind(state: np.ndarray, action: Tuple[float, float],
                   params: Dict, rng: np.random.Generator) -> np.ndarray:
    dt = params["dt"]
    wind = rng.normal(params["wind_vel_mean"], params["wind_gust_sigma"])
    u, phi = action
    #NOTE: rollout assumes instantaneous actuators for speed (same across candidates)
    ns = rk4_step(state, u, phi, params, wind, dt)
    ns[6] = max(0.0, ns[6])
    return ns

def _terminal(state: np.ndarray, params: Dict) -> bool:
    x, z, vx, vz, theta, theta_dot, f = state
    if z <= 0.0:
        return True
    if abs(theta) > 3*params["max_gimbal"] or abs(x) > 1e3:
        return True
    return False

def _shape_reward(state: np.ndarray, action: Tuple[float, float], params: Dict) -> float:
    x, z, vx, vz, theta, theta_dot, f = state
    u, phi = action
    return (
        -0.05 * abs(u)
        - 0.001 * (x**2 + z**2)
        - 0.005 * (vx**2 + vz**2)
        - 0.002 * (theta**2 + theta_dot**2)
    )
