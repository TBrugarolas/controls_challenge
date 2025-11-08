from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simple optimized PID controller
  """
  def __init__(self, params=[0.195, 0.1, -0.053, 0]):
    self.p = params[0]
    self.i = params[1]
    self.d = params[2]
    self.error_integral = params[3]
    self.prev_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = (target_lataccel - current_lataccel)
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error
    return self.p * error + self.i * self.error_integral + self.d * error_diff
