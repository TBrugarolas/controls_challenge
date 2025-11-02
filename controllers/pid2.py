from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A more advanced PID controller
  """
  def __init__(self, params=[0.34182721, 0.16471837, -0.07861072, -0.02190636,  0.13698225, 0.08638133, 0.09460732, -0.12039244]):
    self.p = params[0]
    self.i = params[1]
    self.d = params[2]
    self.dd = params[3]
    self.alpha = params[4]
    self.beta = params[5]
    self.gamma1 = params[6]
    self.gamma2 = params[7]

    self.error_integral = 0
    self.prev_error = 0
    self.prev_prev_error = 0

  def gain_schedule_p(self, v):
    return self.p * (1.0 / (1 + self.alpha * v))

  def gain_schedule_i(self, v):
    return self.i * (1.0 / (1 + self.beta * v))

  def gain_schedule_d(self, v, roll):
    return self.d * (1.0 / (1 + self.gamma1 * v)) * np.exp(-self.gamma2*abs(roll))


  def update(self, target_lataccel, current_lataccel, state, future_plan):
    v = state.v_ego*.1
    roll = state.roll_lataccel*.1
    p = self.gain_schedule_p(v)
    i = self.gain_schedule_i(v)
    d = self.gain_schedule_d(v, roll)
    dd = self.dd

    error = (target_lataccel - current_lataccel)
    error_diff = error - self.prev_error
    error_ddiff = self.prev_error - self.prev_prev_error

    # Clegg integrator:
    if np.sign(error_diff) != np.sign(error_ddiff):
        self.error_integral -= error
    else:
        self.error_integral += error

    self.prev_prev_error = self.prev_error
    self.prev_error = error

    return p * error + i * self.error_integral + d * error_diff + dd * error_ddiff
