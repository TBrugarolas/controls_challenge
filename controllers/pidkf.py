from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  PID + Kalman Filter for estimated lateral acceleration
  """
  def __init__(self):
    # PID gains
    self.p = 0.195
    self.i = 0.100
    self.d = -0.053
    
    # PID state
    self.error_integral = 0
    self.prev_error = 0

    # ---- Kalman Filter Setup ----
    # State: x = [lat_accel, lat_accel_rate]
    self.x_hat = np.array([[0.0], [0.0]])  # initial estimate
    self.P = np.eye(2) * 1.0               # initial covariance
    
    # Discrete-time model (assumed small dt)
    self.dt = 0.05  # <-- this may need tuning based on environment
    self.A = np.array([[1, self.dt],
                       [0, 1]])
    self.B = np.array([[0],
                       [0]])  # no direct control input known
    self.H = np.array([[1, 0]])  # we measure lat_accel directly

    # Noise levels (to tune)
    self.Q = np.eye(2) * 0.5  # model noise
    self.R = np.array([[0.15]]) # measurement noise

  def kalman_update(self, measured_lataccel):
    # Prediction
    self.x_hat = self.A @ self.x_hat
    self.P = self.A @ self.P @ self.A.T + self.Q

    # Measurement update
    y = measured_lataccel - (self.H @ self.x_hat)[0,0]
    S = self.H @ self.P @ self.H.T + self.R
    K = self.P @ self.H.T @ np.linalg.inv(S)
    
    self.x_hat = self.x_hat + K * y
    self.P = (np.eye(2) - K @ self.H) @ self.P

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    # Use Kalman filter to denoise/estimate accel and accel rate
    self.kalman_update(current_lataccel)
    est_lataccel = self.x_hat[0,0]

    # ----- PID Control -----
    error = target_lataccel - est_lataccel
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error

    return self.p * error + self.i * self.error_integral + self.d * error_diff