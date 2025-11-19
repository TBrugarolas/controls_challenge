from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simple PIDD controller
  """
  def __init__(self,):
    self.p = 0.195
    self.i = 0.100
    self.d = -0.053
    self.d2 = self.d * 0.1
    self.error_integral = 0
    self.prev_error = 0
    self.prev_diff_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = (target_lataccel - current_lataccel)
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error
    diff_error_diff = error_diff - self.prev_diff_error
    self.prev_diff_error = error_diff
    return self.p * error + self.i * self.error_integral + self.d * error_diff + self.d2 * diff_error_diff
