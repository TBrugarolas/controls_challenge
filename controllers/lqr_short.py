from . import BaseController
import numpy as np
from tinyphysics import TinyPhysicsModel
from typing import List, Union, Tuple, Dict
from collections import namedtuple

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])
CONTEXT_LENGTH = 20
MAX_DELTA_U = 0.01
DEL_T = 0.1
TAU = 0.3

class Controller(BaseController):
  """
  An LQR controller
  """
  def __init__(self):
    self.Q = np.diag([1, 1, 10, 1])
    self.R = np.array([[1]])
    self.model = TinyPhysicsModel("models/tinyphysics.onnx", debug=False)

    self.state_history = []
    self.action_history = []
    self.current_lataccel_history = []

  @staticmethod
  def TI_LQR(A, B, Q, R, N):
    P = Q
    K_list = []
    for t in reversed(range(N)):
      K_t = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
      K_list.insert(0, K_t)
      P = Q + A.T @ P @ (A - B @ K_t)
    return K_list

  def _predict_next_state(self, state: np.ndarray, u: float) -> np.ndarray:
    """
    Uses TinyPhysicsModel to predict next state given (state, u).
    """
    # Unpack the state vector
    v_ego, a_ego, lataccel, roll_lataccel = state

    v_next = v_ego + a_ego * DEL_T
    a_next = a_ego + (u - a_ego) * DEL_T / TAU

    # Update histories for prediction context
    self.state_history.append(State(roll_lataccel, v_ego, a_ego))
    self.action_history.append(u)
    self.current_lataccel_history.append(lataccel)

    # Prepare model inputs (ensure correct length)
    sim_states = list(self.state_history)[-CONTEXT_LENGTH:]
    actions = list(self.action_history)[-CONTEXT_LENGTH:]
    past_preds = list(self.current_lataccel_history)[-CONTEXT_LENGTH:]

    # Predict next lateral acceleration
    pred = self.model.get_current_lataccel(
      sim_states=sim_states,
      actions=actions,
      past_preds=past_preds
    )

    # Construct predicted next state
    next_state = np.array([v_next, a_next, pred, roll_lataccel])
    return next_state


  def linearize(self, state: np.ndarray, u: float = 0.0, eps: float = .01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numerically linearizes the TinyPhysicsModel around (state, u).
    Returns:
      A: State Jacobian (df/dx)
      B: Input Jacobian (df/du)
    """
    n = len(state)
    A = np.zeros((n, n))
    B = np.zeros((n, 1))

    # Finite differences for A
    for i in range(n):
      dx = np.zeros_like(state)
      dx[i] = eps
      f_plus = self._predict_next_state(state + dx, u)
      f_minus = self._predict_next_state(state - dx, u)
      A[:, i] = (f_plus - f_minus) / (2 * eps)

    # Finite difference for B
    f_plus = self._predict_next_state(state, u + eps)
    f_minus = self._predict_next_state(state, u - eps)
    B[:, 0] = (f_plus - f_minus) / (2 * eps)

    return A, B

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    # state vector
    roll_lataccel, v_ego, a_ego = state
    x = np.array([v_ego, a_ego, current_lataccel, roll_lataccel])
    
    # reference trajectory
    target_lataccel_seq, roll_lataccel_seq, v_ego_seq, a_ego_seq = future_plan
    if len(target_lataccel_seq) == 0:
      target_lataccel_seq = [target_lataccel]
      roll_lataccel_seq = [roll_lataccel]
      v_ego_seq = [v_ego]
      a_ego_seq = [a_ego]      

    r = np.vstack([v_ego_seq[0], a_ego_seq[0], target_lataccel_seq[0], roll_lataccel_seq[0]]).T
    N = len(r)

    # --- Initialize histories if first call ---
    if len(self.state_history) < CONTEXT_LENGTH:
      for _ in range(CONTEXT_LENGTH):
        self.state_history.append(State(roll_lataccel, v_ego, a_ego))
        self.action_history.append(0.0)
        self.current_lataccel_history.append(current_lataccel)
    
    self.state_history.append(State(roll_lataccel, v_ego, a_ego))
    self.current_lataccel_history.append(current_lataccel)

    # Linearize and compute LQR gain sequence
    self.A, self.B = self.linearize(x)
    N = len(r)
    K = self.TI_LQR(self.A, self.B, self.Q, self.R, N)

    u_list = []
    for t in range(N):
      err = (x - r[t])
      u_t = -K[t] @ err
      u_list.append(u_t)
      x = self.A @ x + self.B * u_t

    # Apply only the first control input and record it
    u_out = np.clip(float(u_list[0]),
            self.action_history[-1] - MAX_DELTA_U,
            self.action_history[-1] + MAX_DELTA_U)
    self.action_history.append(u_out)
    return u_out