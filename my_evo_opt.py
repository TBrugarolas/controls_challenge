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


### BASIC EVOLUTION: KEEPING TOP 10, GENERATE THE 90 NEW ###
def sample_offspring(N: int, lambda_: int, rng, xmean, sigma):
    """Generates lambda offspring w/ gaussian noise around current weighted mean."""
    z = rng.standard_normal((N, lambda_))
    arx = xmean + sigma * z
    return arx


def evolve(controller_module,
           params0: List[float],
           model_path: str = "./models/tinyphysics.onnx",
           data_dir: str = "./data",
           carry_over: int = 10,
           pop_size: int = 100,
           stopfitness: float = 50,
           best_weight: float = .5,
           num_segs_per_eval: int = 20,
           verbose: bool = True,
           seed: int = 42,
           log_path: str = "myopt_log_pid2_trial1.csv",
           max_evals: int = 100000):
    rng = np.random.default_rng(seed)
    xmean = np.asarray(params0, dtype=float).reshape(-1, 1)
    N = xmean.shape[0]
    lambda_ = int(pop_size - carry_over)
    sigma = 0.3

    data_path = Path(data_dir)
    files = sorted(data_path.iterdir())[:num_segs_per_eval]
    if len(files) == 0:
        raise RuntimeError(f"No data files found in {data_dir}")

    counteval = 0
    history = {'best_fitness': [], 'mean_fitness': [], 'params': []}
    gen = 0

    log_file = open(log_path, "w", buffering=1)
    log_file.write("generation,evals,best_fitness,mean_fitness,params\n")

    # FIX: initialize xtop and weights
    xtop = [xmean.copy() for _ in range(carry_over)]
    weights = np.ones(carry_over) / carry_over

    try:
        # === Main Loop ===
        while counteval < max_evals:
            gen += 1

            # 1. Sample offspring + Combine with top performers
            arx = sample_offspring(N, lambda_, rng, xmean, sigma)
            arx_list = xtop + [arx[:, k].reshape(N, 1) for k in range(lambda_)]

            # 2. Evaluate offspring
            eval_results = []
            for k in range(len(arx_list)):
                result = evaluate_candidate(arx_list[k].flatten(), files, controller_module, model_path)
                eval_results.append(result)
            arfitness = np.array([r['total_cost'] for r in eval_results])
            counteval += len(arx_list)

            # 3. Sort and recombine
            arindex = np.argsort(arfitness)
            arfitness_sorted = arfitness[arindex]
            arx_sorted = np.hstack(arx_list)[..., arindex]  # make arx_sorted (N, total)
            xtop = [arx_sorted[:, i].reshape(N, 1) for i in range(carry_over)]
            # Recalculate weights: 50% to best, remaining 50% distributed among rest based on closeness to best.
            if carry_over > 1:
                best_fit = arfitness_sorted[0]
                others_fit = arfitness_sorted[1:carry_over]

                rel_scores = np.maximum(0, (others_fit[-1] - others_fit) / (others_fit[-1] - best_fit + 1e-12))
                rel_scores /= rel_scores.sum() if rel_scores.sum() > 0 else 1.0

                w_best = best_weight
                w_rest = best_weight * rel_scores
                weights = np.concatenate([[w_best], w_rest])
            else:
                weights = np.array([1.0])

            weights /= weights.sum()

            xmean = sum(w * x for w, x in zip(weights, xtop))

            # 4. Logging & stopping
            best_fitness = float(arfitness_sorted[0])
            mean_fitness = float(np.mean(arfitness_sorted[:carry_over]))
            history['best_fitness'].append(best_fitness)
            history['mean_fitness'].append(mean_fitness)
            history['params'].append(xmean.flatten().tolist())

            log_file.write(f"{gen},{counteval},{best_fitness:.6e},{mean_fitness:.6e},\"{xmean.flatten().tolist()}\"\n")
            log_file.flush()

            if verbose:
                print(f"Gen {gen:3d}  Eval {counteval:6d}  Best {best_fitness:.6e}  Mean {mean_fitness:.6e}")

            if best_fitness <= stopfitness:
                if verbose:
                    print("Stopping: reached target fitness")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user — partial results saved to log.")
    finally:
        log_file.close()

    best_index = int(np.argmin(arfitness))
    best_params = np.hstack(arx_list)[:, best_index].flatten()
    return best_params


# === Entrypoint ===
if __name__ == "__main__":
    CONTROLLER = "pid2"
    controller_module = importlib.import_module(f'controllers.{CONTROLLER}')

    # Initial parameters for pid2
    P0 = [0.37367301440636197, 0.18557052557463957, -0.0792914668252033, 0.06331065057335294,
          0.26538137215511515, 0.2686367100203415, 0.0991006278077006, -0.20556991574148054]

    best_params = evolve(controller_module, P0,
                         model_path="./models/tinyphysics.onnx",
                         data_dir="./data",
                         verbose=True)
    print("Found best params:", best_params)
