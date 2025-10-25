from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    self.p = 0.195
    self.i = 0.100
    self.d = -0.053
    self.ff = 0.210
    self.error_integral = 0
    self.prev_error = 0
    self.once = True

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = (target_lataccel - current_lataccel)
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error

    # if self.once:
    #   print("STATE:", state)
    #   print("FUTURE PLAN:", future_plan)
    #   self.once = False
    self.FF = 0
    self.alpha = self.ff
    for i in range(len(future_plan.lataccel)):
      self.FF += self.alpha * future_plan.lataccel[i]
      self.alpha /= 5

    return self.p * error + self.i * self.error_integral + self.d * error_diff + self.FF
