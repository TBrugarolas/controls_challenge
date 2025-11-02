from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A basic PID + FF controller
  """
  def __init__(self, params=[0.195, 0.100, -0.053, 0.210]):
    params = [0.1274712696507233, 0.12337218260952829, 0.015426298824837252, 0.13241204858670386]
    self.p = params[0]
    self.i = params[1]
    self.d = params[2]
    self.ff = params[3]
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
      self.alpha /= 2

    return self.p * error + self.i * self.error_integral + self.d * error_diff + self.FF
