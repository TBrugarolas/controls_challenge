from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A Loop Shaping controller
  """
  def __init__(self,):
    self.p = 0.195
    self.i = 0.100
    self.d = -0.053
    self.error_integral = 0
    self.prev_error = 0
    self.model = TinyPhysicsModel("models/tinyphysics.onnx", debug=False)

  def finite_diff_jacobian(self, state, eps=1e-4):
    x = np.array(state, dtype=np.float32)
    n = x.size
    y0 = self.model.predict({"states":x, "tokens":None})
    J = np.zeros((1, n))
    for i in range(n):
        dx = np.zeros_like(x)
        dx[i] = eps
        y1 = self.model.predict({"states":x+dx, "tokens":None})
        J[:, i] = (y1 - y0) / eps
    print(J)
    return J

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = (target_lataccel - current_lataccel)
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error
    return self.p * error + self.i * self.error_integral + self.d * error_diff
