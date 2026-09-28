import os, sys
from rnn import *
from utils import subdirs
from R2D3 import * 
from argparse import ArgumentParser
from setup import *
import importlib 

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
print([base_model_paths])

if args.a == "":
    sortedModels = base_models

else:
    sortedIdxs = np.argsort([base_model.params[args.a] for base_model in base_models])
    sortedModels = [base_models[i] for i in sortedIdxs]

# set up task

_, model_params, task_params, training_params = load_rnn(base_model_paths[0])
cycles = Task.load_task("Cycling", task_params)
inputs = cycles.generate_inputs(NUM_ANGLES)

# set up some tensors
trial_len = task_params["trial_len"]
out_dim = task_params["out_dim"]
trials = torch.vstack([cycles[i][0] for i in range(len(cycles))]).reshape(-1, trial_len, out_dim)
targets = torch.vstack([cycles[i][1] for i in range(len(cycles))]).reshape(-1, trial_len, out_dim)
masks = torch.vstack([cycles[i][2] for i in range(len(cycles))]).reshape(-1, trial_len, out_dim)


def run4model(model):


    rates, out = run_rnn(model, inputs)
    errors = []
    solutions = []

    for rank in range(1, args.r):
        Solution = R2D3SolutionObject._setup(model, inputs, rank=rank, ridge=1e-1, mechanistic_rank=None)
        Solution.r2d3_tube_fit(n_perturbations=4)
        solutions.append(Solution)
        errors.append(Solution.normalised_estimation_error)

    # we still wa
    for idx in range(len(solutions)):
        sol = solutions[idx]
        lr_model = sol.build_LowRankRNN()
        _, out = run_rnn(lr_model, trials)
        err = mse(out, targets, masks)
        if err <= args.t:
            break

    #build the LR model
    lr_model = sol.build_LowRankRNN()

    return sol, errors, lr_model

if __name__ == "__main__":

    for idx in range(len(sortedModels)):
        sol, errors, lr_model = run4model(sortedModels[idx])
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

    