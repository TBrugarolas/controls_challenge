import argparse
import importlib
import numpy as np
import onnxruntime as ort
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import signal
import urllib.request
import zipfile

from io import BytesIO
from collections import namedtuple
from functools import partial
from hashlib import md5
from pathlib import Path
from typing import Optional, List, Union, Tuple, Dict

from controllers import BaseController
from tinyphysics import LataccelTokenizer, TinyPhysicsModel, TinyPhysicsSimulator

# === Constants ===
ACC_G = 9.81
FPS = 10
CONTROL_START_IDX = 100
COST_END_IDX = 500
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0
FUTURE_PLAN_STEPS = FPS * 5  # 5 secs

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])


# === Helper: run simulation ===
def run_rollout(data_path, controller, model_path, debug=False):
    tinyphysicsmodel = TinyPhysicsModel(model_path, debug=debug)
    sim = TinyPhysicsSimulator(tinyphysicsmodel, str(data_path), controller=controller, debug=debug)
    return sim.rollout(), sim.target_lataccel_history, sim.current_lataccel_history


# === Helper: Rosenbrock (for debug) ===
def f_rosenbrock(x: np.ndarray) -> float:
    x = np.asarray(x).flatten()
    if x.size < 2:
        raise ValueError("dimension must be greater than one")
    return 100.0 * np.sum((x[:-1] ** 2 - x[1:]) ** 2) + np.sum((x[:-1] - 1.0) ** 2)


# === Evaluation Helpers ===
def evaluate_candidate(params_vector: np.ndarray,
                       files_subset: List[Path],
                       controller_module,
                       model_path: str) -> Dict:
    """Evaluates one candidate controller over a subset of data files."""
    controller_obj = controller_module.Controller(params_vector.flatten().tolist())
    per_file_results = []
    for f in files_subset:
        rollout, target_hist, current_hist = run_rollout(f, controller_obj, model_path, debug=False)
        per_file_results.append(rollout)
    df = pd.DataFrame(per_file_results)
    return {
        'lataccel_cost': float(df['lataccel_cost'].mean()),
        'jerk_cost': float(df['jerk_cost'].mean()) if 'jerk_cost' in df.columns else float(df.mean().mean()),
        'total_cost': float(df['total_cost'].mean()) if 'total_cost' in df.columns else float(df.mean().mean())
    }


def files_for_index(i: int, sorted_files: List[Path], num_segs_per_eval: int) -> List[Path]:
    return sorted_files[:num_segs_per_eval]
    ### This was stupid, dont change the metric dummy ###
    # """Selects a slice of data files for one evaluation."""
    # start = (num_segs_per_eval * i) % len(sorted_files)
    # end = start + num_segs_per_eval
    # if end <= len(sorted_files):
    #     return sorted_files[start:end]
    # else:
    #     return sorted_files[start:] + sorted_files[:(end % len(sorted_files))]


def sample_offspring(N: int, lambda_: int, rng, xmean, sigma, B, D):
    """Generates lambda offspring around current mean."""
    z = rng.standard_normal((N, lambda_))
    y = B @ (D[:, None] * z)
    if np.isscalar(sigma):
        arx = xmean + sigma * y
    else:
        s_arr = np.asarray(sigma).reshape(N, 1)
        arx = xmean + s_arr * y
    return arx, z, y


# === Main CMA-ES ===
def CMA_ES(controller_module,
           params0: List[float],
           model_path: str = "./models/tinyphysics.onnx",
           data_dir: str = "./data",
           population_factor: float = 3.0,
           max_evals: Optional[int] = None,
           stopfitness: float = 50,
           num_segs_per_eval: int = 20,
           verbose: bool = True,
           seed: int = 42,
           log_path: str = "cma_log_pid2_trial1.csv") -> Tuple[np.ndarray, Dict]:
    """
    CMA-ES optimizer with persistent logging of progress each generation.
    """

    rng = np.random.default_rng(seed)
    xmean = np.asarray(params0, dtype=float).reshape(-1, 1)
    N = xmean.shape[0]
    sigma = 0.3

    # CMA-ES strategy parameters
    lambda_ = 4 + int(np.floor(population_factor * np.log(N)))
    mu = lambda_ // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mueff = (np.sum(weights) ** 2) / np.sum(weights ** 2)

    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
    cs = (mueff + 2) / (N + mueff + 5)
    c1 = 2 / ((N + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1.0 / mueff) / ((N + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs

    pc = np.zeros((N, 1))
    ps = np.zeros((N, 1))
    B = np.eye(N)
    D = np.ones(N)
    C = B @ np.diag(D ** 2) @ B.T
    invsqrtC = B @ np.diag(D ** -1) @ B.T
    eigeneval = 0
    chiN = np.sqrt(N) * (1.0 - 1.0 / (4 * N) + 1.0 / (21 * (N ** 2)))

    data_path = Path(data_dir)
    sorted_files = sorted(data_path.iterdir())
    if len(sorted_files) == 0:
        raise RuntimeError(f"No data files found in {data_dir}")

    counteval = 0
    history = {'best_fitness': [], 'mean_fitness': [], 'params': []}
    gen = 0

    if max_evals is None:
        max_evals = int(1e3 * (N ** 2))

    local_test = (controller_module is None)

    # --- Setup log file ---
    log_file = open(log_path, "w", buffering=1)  # line-buffered
    log_file.write("generation,evals,best_fitness,mean_fitness,params\n")

    try:
        # === Main Loop ===
        while counteval < max_evals:
            gen += 1

            # 1. Sample offspring
            arx, arz, ary = sample_offspring(N, lambda_, rng, xmean, sigma, B, D)
            arx_list = [arx[:, k].reshape(N, 1) for k in range(lambda_)]

            # 2. Evaluate offspring
            file_indices = [(gen - 1) * lambda_ + k for k in range(lambda_)]
            file_subsets = [files_for_index(idx, sorted_files, num_segs_per_eval) for idx in file_indices]

            if local_test:
                arfitness = np.array([f_rosenbrock(x.flatten()) for x in arx_list])
                eval_results = [{'total_cost': v} for v in arfitness]
            else:
                eval_results = []
                for k in range(lambda_):
                    result = evaluate_candidate(arx_list[k].flatten(), file_subsets[k], controller_module, model_path)
                    eval_results.append(result)
                arfitness = np.array([r['total_cost'] for r in eval_results])

            counteval += lambda_

            # 3. Sort and recombine
            arindex = np.argsort(arfitness)
            arfitness_sorted = arfitness[arindex]
            arx_sorted = arx[:, arindex]
            xold = xmean.copy()
            xmean = (arx_sorted[:, :mu] @ weights.reshape(-1, 1)).reshape(N, 1)

            # 4. Update evolution paths
            if np.isscalar(sigma):
                zmean = (invsqrtC @ (xmean - xold)) / sigma
            else:
                s_arr = np.asarray(sigma).reshape(N, 1)
                zmean = (invsqrtC @ (xmean - xold)) / s_arr

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * zmean
            norm_ps = np.linalg.norm(ps)
            hsig = (norm_ps / np.sqrt(1.0 - (1.0 - cs) ** (2.0 * counteval / lambda_)) / chiN) < (1.4 + 2.0 / (N + 1.0))
            pc = (1 - cc) * pc + (hsig.astype(float) * np.sqrt(cc * (2 - cc) * mueff) *
                                  ((xmean - xold) / (sigma if np.isscalar(sigma) else s_arr)))

            # 5. Adapt covariance
            artmp = (1.0 / (sigma if np.isscalar(sigma) else s_arr)) * (arx_sorted[:, :mu] - np.tile(xold, (1, mu)))
            C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + (1.0 - hsig) * cc * (2 - cc) * C) + cmu * (artmp @ np.diag(weights) @ artmp.T)

            # 6. Adapt step-size
            sigma = (sigma if np.isscalar(sigma) else s_arr) * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1.0))
            if not np.isscalar(sigma) and np.asarray(sigma).size == 1:
                sigma = float(sigma)

            # 7. Update B, D
            if counteval - eigeneval > (lambda_ / (c1 + cmu) / N / 10.0):
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                D2, B = np.linalg.eigh(C)
                D2[D2 < 0] = 1e-12
                D = np.sqrt(D2)
                invsqrtC = B @ np.diag(1.0 / D) @ B.T

            # 8. Logging & stopping
            best_fitness = float(arfitness_sorted[0])
            mean_fitness = float(np.mean(arfitness_sorted))
            history['best_fitness'].append(best_fitness)
            history['mean_fitness'].append(mean_fitness)
            history['params'].append(xmean.flatten().tolist())

            # Write to log file
            log_file.write(f"{gen},{counteval},{best_fitness:.6e},{mean_fitness:.6e},\"{xmean.flatten().tolist()}\"\n")
            log_file.flush()

            if verbose:
                print(f"Gen {gen:3d}  Eval {counteval:6d}  Best {best_fitness:.6e}  Mean {mean_fitness:.6e}")

            if mean_fitness <= stopfitness:
                if verbose:
                    print("Stopping: reached target fitness")
                break
            if np.max(D) > 1e7 * np.min(D):
                if verbose:
                    print("Stopping: condition number too large")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user — partial results saved to log.")
    finally:
        log_file.close()

    best_index = int(np.argmin(arfitness))
    best_params = arx[:, best_index].flatten()
    return best_params, history



# === Entrypoint ===
if __name__ == "__main__":
    CONTROLLER = "pid2"
    controller_module = importlib.import_module(f'controllers.{CONTROLLER}')
    # Initial parameters for pid2
    P0 = [0.37367301440636197, 0.18557052557463957, -0.0792914668252033, 0.06331065057335294, 0.26538137215511515, 0.2686367100203415, 0.0991006278077006, -0.20556991574148054]
    # Initial parameters for pidff
    # P0 = [0.195, 0.100, -0.053, 0.210]

    best_params, hist = CMA_ES(controller_module, P0,
                               model_path="./models/tinyphysics.onnx",
                               data_dir="./data",
                               population_factor=3.0,
                               verbose=True)
    print("Found best params:", best_params)
