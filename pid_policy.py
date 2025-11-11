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