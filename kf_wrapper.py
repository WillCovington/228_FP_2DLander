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
        '''Initialize the Extended Kalman Filter (EKF) for the 2D rocket lander.
        Builds the measurement noise covariance R from sensor sigmas and
        the process noise covariance Q from the EKFConfig.
        Initializes the state mean and covariance.
        @param: (dict): Rocket and environment parameters, including sensor noise,
        dynamics constants, and wind mean.
        @param: cfg (EKFConfig): Confifugration for the EKF numerical Jacobian step
        and process noise levels. 
        '''
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
        '''Reset the EKF state estimate and covariance.
        @param: z0: np.ndarray: Initial state estimate (7-dim state vector)
        @param: P0: nd.ndarray (optional): Initial covariance matrix. If none,
        default diagonal covariance matrix is used. Also resets the stored last action used for prediction.
        '''
        self.mu = z0.astype(float).copy()
        self.Sigma = P0.copy() if P0 is not None else np.diag([1.0,1.0,1.0,1.0,0.5,0.5,5.0])
        self.last_action = (0.0, 0.0)

    def f(self, x: np.ndarray, a: Tuple[float, float]) -> np.ndarray:
        '''Rocket motion model used by the EKF predict step.
        @param: x (np.ndarray): Current state.
        @param: a (float, float): Control input (throttle, gimbal angle).
        @return (np.ndarray): Predicted next state after one timestep, integrated using RK4.
        '''
        #one-step motion using rk4; use mean wind for predict (NOTE: zero-mean gust absorbed by Q)
        wind = self.p["wind_vel_mean"]
        dt = self.p["dt"]
        u, phi = a
        nx = rk4_step(x, u, phi, self.p, wind, dt)
        nx[6] = max(0.0, nx[6])  #we assume fuel is nonnegative
        return nx

    def F_jacobian(self, x: np.ndarray, a: Tuple[float, float]) -> np.ndarray:
        ''' Compute the Jacobian of the motion model f with respect to the state.
        @param: x (np.ndarray): State around which to linearize.
        @param: a (float, float): Control input at which the Jacobian is evaluated.
        @return (np.ndarray): The 7x7 Jacobian matrix F.
        '''
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
        '''EKF prediction step: propagate state estimate and covariance forward. 
        Computes Jacobian F, predicts next state via nonlinear model f(), updates
        covariance via F \sum transpose(F) + Q, and wraps angle to [-pi, pi].
        @param: a_prev (float, float): Control input applied at the previous timestep
        '''
        F = self.F_jacobian(self.mu, a_prev) #compute Jacobian F
        self.mu = self.f(self.mu, a_prev)
        self.mu[4] = (self.mu[4] + np.pi) % (2*np.pi) - np.pi
        Qeff = self.Q * (3.0 if self.mu[1] < 30.0 else 1.0)
        self.Sigma = F @ self.Sigma @ F.T + Qeff

    def update(self, z: np.ndarray):
        '''EKF update step: fuse the noisy observation with the predicted state.
        @param: z (np.ndarray): noisy sensor measurement (direct observation of all state components).
        Uses H = I since observations directly correspond to the state, computes innovation, innovation 
        covariance, and Kalman gain, applies a chi-square NIS gate to reject outlier measurements,
        updates both state mean and covariance, and ensures angle remains wrapped to [-pi, pi].
        '''
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
    '''Guidance policy that combines EKF state estimation with rollout control.
    Maintains a persistent EKF instance across calls. Predicts using the previous action,
    updates using the new noisy observation, runs the Monte-Carlo rollout policy on the
    filtered state, and stores the selected action for the next predict step.
    @param: obs (np.ndarray): Current noisy observations from sensors
    @param info (dict) Simulation metadata including parameters and timestep index.
    @return (float, float): selected (throttle, gimbal) command.
    '''
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