import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
from rocket_simulator import rk4_step  #motion model
from pid_policy import rollout_guidance_policy

'''Contains the Kalman filter for the rollout policy. This class essentially
acts as a wrapper class for the rollout policy. The Kalman
filter estimates the rocket's true state from noisy sensor readings
and an imperfect motion model. At each time step, it uses the rocket dynamics 
(from rk4_step) and the last action to estimate what the next state should be.
Then, it corrects that prediction using the new noisy observation, weighting
each by their uncertainty. The result is a *filtered best estimate* of the state
which is used by the rollout policy for decision making instead of raw, noisy
data.
'''

@dataclass
class EKFConfig:
    eps: float = 1e-4  #finite-diff step for Jacobian
    #process noise (tune if needed; higher on velocities/ang rates to absorb wind/lag mismatch)
    q_x: float = 1e-4
    q_z: float = 1e-4
    q_vx: float = 5e-2
    q_vz: float = 5e-2
    q_theta: float = 5e-3
    q_omega: float = 5e-2
    q_fuel: float = 1e-5

class EKF2DLander:
    def __init__(self, params: Dict, cfg: EKFConfig = EKFConfig()):
        self.p = params
        self.cfg = cfg
        self.n = 7
        self.I = np.eye(self.n)
        #build constant R from sensor sigmas in params as follows:
        Rdiag = np.array([
            self.p["obs_pos_sigma"]**2,
            self.p["obs_pos_sigma"]**2,
            self.p["obs_vel_sigma"]**2,
            self.p["obs_vel_sigma"]**2,
            self.p["obs_angle_sigma"]**2,
            self.p["obs_angvel_sigma"]**2,
            self.p["obs_fuel_sigma"]**2,
        ], dtype=float)
        self.R = np.diag(Rdiag)
        self.Q = np.diag([
            self.cfg.q_x, self.cfg.q_z, self.cfg.q_vx, self.cfg.q_vz,
            self.cfg.q_theta, self.cfg.q_omega, self.cfg.q_fuel
        ])
        self.reset(np.zeros(self.n), np.eye(self.n))
        self.last_action = (0.0, 0.0)

    def reset(self, z0: np.ndarray, P0: np.ndarray = None):
        self.mu = z0.astype(float).copy()
        self.Sigma = P0.copy() if P0 is not None else np.diag([1.0,1.0,1.0,1.0,0.5,0.5,5.0])
        self.last_action = (0.0, 0.0)

    def f(self, x: np.ndarray, a: Tuple[float, float]) -> np.ndarray:
        #one-step motion using rk4; use mean wind for predict (NOTE: zero-mean gust absorbed by Q)
        wind = self.p["wind_vel_mean"]
        dt = self.p["dt"]
        u, phi = a
        nx = rk4_step(x, u, phi, self.p, wind, dt)
        nx[6] = max(0.0, nx[6])  #we assume fuel is nonnegative
        return nx

    def F_jacobian(self, x: np.ndarray, a: Tuple[float, float]) -> np.ndarray:
        n, eps = self.n, self.cfg.eps
        F = np.zeros((n, n), dtype=float)
        fx = self.f(x, a)
        for i in range(n):
            dx = np.zeros(n); dx[i] = eps
            f_plus  = self.f(x + dx, a)
            f_minus = self.f(x - dx, a)
            F[:, i] = (f_plus - f_minus) / (2.0 * eps)
        return F

    def predict(self, a_prev: Tuple[float, float]):
        F = self.F_jacobian(self.mu, a_prev)
        self.mu = self.f(self.mu, a_prev)
        self.mu[4] = (self.mu[4] + np.pi) % (2*np.pi) - np.pi
        Qeff = self.Q * (3.0 if self.mu[1] < 30.0 else 1.0)
        self.Sigma = F @ self.Sigma @ F.T + Qeff

    def update(self, z: np.ndarray):
        #H = I for direct noisy observations
        S = self.Sigma + self.R
        Sinv = np.linalg.inv(S)
        y = z - self.mu
        y[4] = (y[4] + np.pi) % (2*np.pi) - np.pi
        nis = float(y.T @ Sinv @ y)
        if nis > 18.5:
            self.Sigma = self.Sigma + 0.1 * self.I
            return
        K = self.Sigma @ Sinv
        self.mu = self.mu + K @ y
        self.mu[4] = (self.mu[4] + np.pi) % (2*np.pi) - np.pi
        self.Sigma = (self.I - K) @ self.Sigma

#keep one EKF instance between calls (per process). Reset when step==0.
def kf_rollout_guidance_policy(obs: np.ndarray, info: Dict) -> Tuple[float, float]:
    params = info["params"]
    step = info.get("step", 0)

    #lazy-create EKF and stash on the function object
    if not hasattr(kf_rollout_guidance_policy, "_ekf") or step == 0:
        kf_rollout_guidance_policy._ekf = EKF2DLander(params)
        kf_rollout_guidance_policy._ekf.reset(obs, np.diag([10,10,5,5,1,1,20]))

    ekf = kf_rollout_guidance_policy._ekf

    #Step 1: predict with last action (a_{t-1}). At step=0 this is (0,0)
    ekf.predict(ekf.last_action)

    #Step 2: update with current observation
    ekf.update(obs)
    mu = ekf.mu  # state estimate

    #Step 3: plan using the estimated state (reuse your rollout as-is)
    #We pass the same info dict so rollout uses the same params.
    action = rollout_guidance_policy(mu, info) #call rollout policy

    #Step 4: store action for next predict and return it
    ekf.last_action = action
    return action