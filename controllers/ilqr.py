from . import BaseController
import numpy as np
from tinyphysics import TinyPhysicsModel
from typing import List, Union, Tuple, Dict
from collections import namedtuple

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])
CONTEXT_LENGTH = 20
MAX_ACC_DELTA = 0.1
DEL_T = 0.1
TAU = 0.3

class Controller(BaseController):
    """
    An LQR controller
    """
    def __init__(self):
        self.Q = np.diag([1, 1, 50, 1])
        self.R = np.array([[100]])
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

    @staticmethod
    def TV_LQR(A_list, B_list, Q, R):
        N = len(A_list)
        P_list = [None] * N
        K_list = [None] * N
        P_list[N-1] = Q

        for t in reversed(range(N - 1)):
            A, B = A_list[t], B_list[t]
            P_next = P_list[t+1]
            # Equations
            K_list[t] = np.linalg.inv(R + B.T @ P_next @ B) @ B.T @ P_next @ A
            P_list[t] = Q + A.T @ P_next @ A - A.T @ P_next @ B @ K_t
        return K_list, P_list
    
    def iLQR(self, x0: np.ndarray, r_seq: np.ndarray, u_init=None, max_iter=10, eps=0.01):
        N = len(r_seq)
        n = len(x0)
        
        # Initialize control sequence
        if u_init is None:
            u_seq = [0.0 for _ in range(N)]
        else:
            u_seq = list(u_init)
        
        for iteration in range(max_iter):
            # --- Forward rollout ---
            x_seq = [x0]
            for t in range(N):
                x_next = self._predict_next_state(x_seq[-1], u_seq[t])
                x_seq.append(x_next)
            
            # --- Linearize along trajectory ---
            A_seq, B_seq = [], []
            for t in range(N):
                A_t, B_t = self.linearize(x_seq[t], u_seq[t], eps=eps)
                A_seq.append(A_t)
                B_seq.append(B_t)
            
            # --- Backward pass ---
            V_x = np.zeros(n)
            V_xx = self.Q
            K_seq = []
            k_seq = []
            for t in reversed(range(N)):
                A, B = A_seq[t], B_seq[t]
                Q_x = self.Q @ (x_seq[t] - r_seq[t]) + A.T @ V_x
                Q_u = self.R * u_seq[t] + B.T @ V_x
                Q_xx = self.Q + A.T @ V_xx @ A
                Q_ux = B.T @ V_xx @ A
                Q_uu = self.R + B.T @ V_xx @ B
                
                # Compute feedback gains
                K_t = -np.linalg.inv(Q_uu) @ Q_ux
                k_t = -np.linalg.inv(Q_uu) @ Q_u
                
                # Update Value function
                V_x = Q_x + K_t.T @ Q_uu @ k_t
                V_xx = Q_xx + K_t.T @ Q_uu @ K_t
                
                K_seq.insert(0, K_t)
                k_seq.insert(0, k_t)
            
            # --- Forward update of controls ---
            x_new = x0.copy()
            for t in range(N):
                u_seq[t] += (k_seq[t] + K_seq[t] @ (x_new - x_seq[t]))[0,0]
                x_new = self._predict_next_state(x_new, u_seq[t])
        
        # Return first control
        return float(u_seq[0])


    def _predict_next_state(self, state: np.ndarray, u: float) -> np.ndarray:
        """
        Uses TinyPhysicsModel to predict next state given (state, u).
        """
        # Unpack the state vector
        v_ego, a_ego, lataccel, roll_lataccel = state

        # v_next = v_ego + a_ego * DEL_T
        # a_next = a_ego + (u - a_ego) * DEL_T / TAU

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
        next_state = np.array([v_ego, a_ego, pred, roll_lataccel])
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

        r = np.vstack([v_ego_seq, a_ego_seq, target_lataccel_seq, roll_lataccel_seq]).T
        N = len(r)

        # --- Initialize histories if first call ---
        if len(self.state_history) < CONTEXT_LENGTH:
            for _ in range(CONTEXT_LENGTH):
                self.state_history.append(State(roll_lataccel, v_ego, a_ego))
                self.action_history.append(0.0)
                self.current_lataccel_history.append(current_lataccel)

        self.state_history.append(State(roll_lataccel, v_ego, a_ego))
        self.current_lataccel_history.append(current_lataccel)

        # Apply only the first control input and record it
        u_out = float(self.iLQR(x, r, max_iter=3))
        self.action_history.append(u_out)
        return u_out