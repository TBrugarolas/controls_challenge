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
from typing import List, Union, Tuple, Dict
from tqdm.contrib.concurrent import process_map

from controllers import BaseController
from tinyphysics import LataccelTokenizer, TinyPhysicsModel, TinyPhysicsSimulator

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
def run_rollout(data_path, controller, model_path, debug=False):
  tinyphysicsmodel = TinyPhysicsModel(model_path, debug=debug)
  sim = TinyPhysicsSimulator(tinyphysicsmodel, str(data_path), controller=controller, debug=debug)
  return sim.rollout(), sim.target_lataccel_history, sim.current_lataccel_history

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--controller", default='pid')
  parser.add_argument("--p", type=float, default=0.195)
  parser.add_argument("--i", type=float, default=0.100)
  parser.add_argument("--d", type=float, default=-0.053)
  parser.add_argument("--num_segs", type=int, default=100)
  args = parser.parse_args()

  model_path = "./models/tinyphysics.onnx"
  data_path = Path("./data" )
  controller = importlib.import_module(f'controllers.{args.controller}').Controller()
  controller.p = args.p
  controller.i = args.i
  controller.d = args.d

  if args.controller == 'pidkf':
    for i in [0.1, 0.5, 1, 5]:
      for j in [0.1, 0.125, 0.15]:
        controller.Q = np.eye(2) * i
        controller.R = np.array([[j]])

        run_rollout_partial = partial(run_rollout, controller=controller, model_path=model_path, debug=False)
        files = sorted(data_path.iterdir())[:args.num_segs]
        results = process_map(run_rollout_partial, files, max_workers=16, chunksize=10)
        costs = [result[0] for result in results]
        costs_df = pd.DataFrame(costs)
        print(f"Q: {i}, R: {j}")
        print(f"Average lataccel_cost: {np.mean(costs_df['lataccel_cost']):>6.4}, ")
        print(f"Average jerk_cost: {np.mean(costs_df['jerk_cost']):>6.4},")
        print(f"Average total_cost: {np.mean(costs_df['total_cost']):>6.4}")
        
  elif args.controller == 'pidff':
    for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
      controller.ff = i

      run_rollout_partial = partial(run_rollout, controller=controller, model_path=model_path, debug=False)
      files = sorted(data_path.iterdir())[:args.num_segs]
      results = process_map(run_rollout_partial, files, max_workers=16, chunksize=10)
      costs = [result[0] for result in results]
      costs_df = pd.DataFrame(costs)
      print(f"FF: {i}")
      print(f"Average lataccel_cost: {np.mean(costs_df['lataccel_cost']):>6.4}, ")
      print(f"Average jerk_cost: {np.mean(costs_df['jerk_cost']):>6.4},")
      print(f"Average total_cost: {np.mean(costs_df['total_cost']):>6.4}")
  
  else:
    run_rollout_partial = partial(run_rollout, controller=controller, model_path=model_path, debug=False)
    files = sorted(data_path.iterdir())[:args.num_segs]
    results = process_map(run_rollout_partial, files, max_workers=16, chunksize=10)
    costs = [result[0] for result in results]
    costs_df = pd.DataFrame(costs)
    print(f"Average lataccel_cost: {np.mean(costs_df['lataccel_cost']):>6.4}, ")
    print(f"Average jerk_cost: {np.mean(costs_df['jerk_cost']):>6.4},")
    print(f"Average total_cost: {np.mean(costs_df['total_cost']):>6.4}")