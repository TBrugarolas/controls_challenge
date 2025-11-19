from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simple PIID controller w/ some integral windup correction
  """
  def __init__(self,):
    self.p = 0.195
    self.i = 0.100
    self.i2 = self.i / 1.5
    self.d = -0.053
    self.error_integral = 0
    self.int_error_integral = 0
    self.prev_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = (target_lataccel - current_lataccel)
    self.error_integral += error
    if error < .5 and error > -.5:
        self.error_integral = 0
    self.int_error_integral += self.error_integral
    if error < .01 and error > -.01:
        self.int_error_integral = 0
    error_diff = error - self.prev_error
    self.prev_error = error
    return self.p * error + self.i * self.error_integral + self.d * error_diff + self.i2 * self.int_error_integral
