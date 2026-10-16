from rnn import RNN, save_rnn
from train import TrainingLogger, train_rnn
from default_params import default_params
from Tasks import cycling
from Tasks.TaskManager import Task
import os
from argparse import ArgumentParser
from tqdm.auto import tqdm

parser = ArgumentParser()
parser.add_argument("--d", 
                    type=str, 
                    help="rel_path of directory in which to store results"
                    )
parser.add_argument("--scales",
                    type=float,
                    nargs="+",
                    help="list of input weight scale")

args = parser.parse_args()
print(args)

model_params, task_params, training_params = default_params()
num_models = 10 # this is per w_in_scale

cycles = Task.load_task("Cycling", task_params) # set up task

model_params["w_in_scale"] = 1 
training_params["lr"] = .001
training_params["n_epochs"] = 10000

os.makedirs(args.d, exist_ok=True)

if __name__ == "__main__":

    for w in tqdm(args.scales, unit="scale"):
        model_params["w_in_scale"] = w

        for idx in tqdm(range(num_models), unit="model"): 
             
            model = RNN(model_params)
            logger = TrainingLogger()
            logger.add_gradients()
            logger.add_parameters()

            losses = train_rnn(model, 
                            cycles, 
                            training_params,
                            logger=logger
                            )
            
            model_path = os.path.join(args.d, f"{idx}")
            save_rnn(model_path,
                    model,
                    model_params,
                    task_params,
                    training_params
                    )
            logger.save(model_path)
