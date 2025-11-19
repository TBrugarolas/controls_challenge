from . import BaseController
import numpy as np

Ku = 0.500
Tu = 20.0 / 11.5
control_type = ["P", "PI", "PD", "PID", "Pessen", "small overshoot", "no overshoot"]
Kp = np.array([0.5, 0.45, 0.8, 0.6, 0.7, 1.0/3.0, 0.2]) * Ku
Ti = np.array([0.01, 5.0/6.0, 0.01, 0.5, 0.3, 0.5, 0.5]) * Tu
Td = np.array([0.0, 0.0, 0.125, 0.125, 0.15, 1.0/3.0, 1.0/3.0]) * Tu
Ki = np.array([0.0, 0.54, 0.0, 1.2, 1.75, 2.0/3.0, 0.4]) * (Kp / Ti)
Kd = np.array([0.0, 0.0, 0.1, 0.075, 0.105, 1.0/9.0, 2.0/30.0]) * (Kp * Ti)

mode = 6

class Controller(BaseController):
  """
  A simple PID controller tuned via Ziegler Nichols method
  """
  def __init__(self,):
    self.p = Kp[mode]
    self.i = Ki[mode]
    self.d = Kd[mode]
    self.error_integral = 0
    self.prev_error = 0
    print(control_type[mode], self.p, self.i, self.d)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = (target_lataccel - current_lataccel)
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error
    return self.p * error + self.i * self.error_integral + self.d * error_diff