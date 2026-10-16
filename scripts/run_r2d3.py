import os, sys
from rnn import *
from utils import subdirs
from R2D3 import * 
from argparse import ArgumentParser
from setup import *
import importlib 
from tqdm.auto import tqdm


parser = ArgumentParser()
parser.add_argument("--r", 
                    type=int, 
                    default=10, 
                    help="maximal rank at which to perform the fit"
                )

parser.add_argument("--t", 
                    type=float,
                    default=.05,
                    help="loss tolerance at which to accept model"
                )

parser.add_argument("--p",
                    type=str,
                    help="directory from which to import models"
                    )

parser.add_argument("--s",
                    type=str,
                    default="",
                    help="selector for which files to import"
                    )

parser.add_argument("--a",
                    type=str,
                    default="",
                    help="which attribute to sort by")

parser.add_argument("--module",
                    type=str,
                    default="cycling"
                    )

args = parser.parse_args()

importlib.import_module(args.module)

base_model_paths = list(subdirs(args.p, start=args.s))
base_models = [load_rnn(path)[0] for path in base_model_paths]

if args.a == "":
    sortedModels = base_models

else:
    sortedIdxs = np.argsort([base_model.params[args.a] for base_model in base_models])
    sortedModels = [base_models[i] for i in sortedIdxs]

# set up task
_, model_params, task_params, training_params = load_rnn(base_model_paths[0])
cycles = Task.load_task("Cycling", task_params)
inputs = cycles.generate_inputs(NUM_ANGLES)

# set up some tensorsxs
trial_len = task_params["trial_len"]
out_dim = task_params["out_dim"]
trials = torch.vstack([cycles[i][0] for i in range(len(cycles))]).reshape(-1, trial_len, out_dim)
targets = torch.vstack([cycles[i][1] for i in range(len(cycles))]).reshape(-1, trial_len, out_dim)
masks = torch.vstack([cycles[i][2] for i in range(len(cycles))]).reshape(-1, trial_len, out_dim)


if __name__ == "__main__":

    for idx in tqdm(range(len(sortedModels)), unit="model"):

        sol, errors, lr_model = R2D3SolutionObject.run4model(sortedModels[idx], 
                                                             inputs,
                                                             targets,
                                                             masks,
                                                             trials,
                                                             max_rank=args.r,
                                                             tol=args.t
                                                             )
        model_path = base_model_paths[idx]
        outPath = os.path.join(model_path, "r2d3_fit")

        save_rnn(outPath,
                 lr_model,
                 lr_model.params,
                 task_params,
                 training_params
                 )
        
        np.savez(os.path.join(
                    outPath, 
                    "errors"
                    ),
                    errors
                    )
        
        # need to unhitch this for pickle
        sol.model = None
        np.savez(os.path.join(
                    outPath, 
                    "solution"
                    ),
                    sol
                    )

        
        