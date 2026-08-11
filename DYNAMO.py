import torch, numpy as np
import torch.nn as nn
from rnn import *
from utils import *
import time

class DynamoRNN(nn.Module):

	"""
	DYNAmical MOdel Embedding for Rnns (cf. Cotler, Susillo et al 2025)
	"""

	def __init__(
			self,
			params, 
			base_models,
			embedding_dim,
			state_map_hidden_size
	):
		
		self.input_size = params["input_size"] + embedding_dim 
		self.embedding_dim = embedding_dim
		self.num_models = len(base_models)
		self.state_map_hidden_size = state_map_hidden_size

		super().__init__()

		#THETAS: embed n models in k dimensions
		self.embeddings = nn.Embedding(
			self.num_models,
			embedding_dim
			)
		
		nn.init.zeros_(self.embeddings.weight)

		"""
		ATTENTION: here's the only major departure from the paper.
		Although it's not clear how they implement it, it seems
		clear that they intend the maps V_i to be some sort of 
		isomorphism between the manifolds produced by the individual
		networks. No idea how to implement that, so we're just gonna
		leverage functional approximation and convert the Vs into
		FFNs and hope for the best.

		We'll keep this simple. It will probably be the most
		elegant if this could itself be made an attribute
		that could be set when instantiating the class. But, again,
		let's not worry about that. Single-layer FFN with tanh activation
		and linear output layer (with or without bias). Constant.
		Hope for the best. 
		That's the plan.


		"""
		
		base_model_sizes = [
							base_model.params["recurrent_size"] 
					  		for base_model in base_models
						]


		self.state_maps = nn.ModuleList([
			nn.Sequential(
				nn.Linear(params["recurrent_size"],
							self.state_map_hidden_size),
				nn.Tanh(),
				nn.Linear(self.state_map_hidden_size,
			   				base_model_size
					)
			 ) for base_model_size in base_model_sizes
		])

		self.meta_params = params.copy()
		self.meta_params["input_size"] = params["input_size"] + embedding_dim

		self.meta_model = RNN(self.meta_params)

	@property 
	def rnn(self):
		return self.meta_model.rnn
	
	def forward(self, input_, model_idx, init_state=None):
		
		batch_size, trial_len, _, = input_.shape

		theta = self.embeddings.weight[model_idx]
		#reshape so theta has the same shape as input
		theta = theta[None, None, :].expand(
            batch_size,
            trial_len,
            -1
        )

		meta_input = torch.cat([input_, theta], dim=-1)
		meta_states, meta_outputs = (
							self.meta_model(
								meta_input,
								init_state=init_state
								)
		)
		mapped_states = self.state_maps[model_idx](meta_states)

		return mapped_states, meta_outputs
	


"""
Could be trained with the original training loop, but that just gets messier and messier. 
"""

def make_dynamo_task(inputs, base_models):

    hidden_targets = []
    out_targets = []

    for model_idx, base_model in enumerate(base_models):
        rates_base, out_base = run_rnn(base_model, inputs)
        hidden_targets.append(rates_base)
        out_targets.append(out_base)

    out_targets = torch.from_numpy(
                        np.vstack(out_targets)
    ).reshape(len(base_models), *out_base.shape)

    hidden_targets = torch.from_numpy(
                        np.vstack(hidden_targets)
    ).reshape(len(base_models), *rates_base.shape)

    return hidden_targets, out_targets

def train_dynamo(
    model,
    inputs,
    target_rates,
    target_outputs,
    n_epochs=100000,
    lr=1e-3,
    lambda_output=1.0,
):
    
    start_time = time.time()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    losses = []
    model.train()

    for epoch in range(n_epochs):

        hidden_loss_epoch = .0
        output_loss_epoch = .0

        for model_idx in range(model.num_models):

            rates_target = target_rates[model_idx]
            outputs_target = target_outputs[model_idx]


            rates_pred, outputs_pred = model(
                inputs,
                model_idx
            )

            hidden_loss = torch.mean(
                (rates_pred - rates_target) ** 2
            )

            output_loss = torch.mean(
                (outputs_pred - outputs_target) ** 2
            )

            loss = (
                hidden_loss
                + lambda_output * output_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            hidden_loss_epoch += hidden_loss
            output_loss_epoch += output_loss


        hidden_loss_epoch /= model.num_models
        output_loss_epoch /= model.num_models

        loss = hidden_loss_epoch + lambda_output * output_loss_epoch 
        losses.append(loss.item())

        print(
			f'epoch {epoch+1 :d} / {n_epochs}\t',
			f'time={time.time()-start_time :.5f}s\t'
			f'task_loss={loss :.5f}',
            f'hidden_loss={hidden_loss :.5f}',
            f'output_loss={output_loss : .5f}',
			end="\r"
			)
            

    model.eval()

    return losses

