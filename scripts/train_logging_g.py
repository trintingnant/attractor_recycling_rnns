from rnn import RNN, save_rnn
from train import TrainingLogger, train_rnn
from default_params import default_params
from Tasks import cycling
from Tasks.TaskManager import Task
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--d", type=str, help="rel_path of directory in which to store results")
args = parser.parse_args()

model_params, task_params, training_params = default_params()
num_models = 30
cycles = Task.load_task("Cycling", task_params) # set up task

model_params["spectral_radius"] = 0.9
model_params["recurrent_size"] = 256
training_params["lr"] = .00001
training_params["n_epochs"] = 10000

os.makedirs(args.d, exist_ok=True)

if __name__ == "__main__":

    for idx in range(num_models):
        if idx >= int(num_models // 2):
            model_params["spectral_radius"] = 1.2

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


