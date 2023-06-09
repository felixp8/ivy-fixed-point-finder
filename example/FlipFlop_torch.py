'''
FlipFlop_torch.py
Written for Python 3.6.9 and Pytorch 1.12.1
@ Matt Golub, June 2023
Please direct correspondence to mgolub@cs.washington.edu
'''

import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

PATH_TO_FIXED_POINT_FINDER = '../'
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
from FixedPoints import FixedPoints
from plot_utils import plot_fps

from FlipFlopData import FlipFlopData
from torch_utils import get_device

class FlipFlopDataset(Dataset):

	def __init__(self, data):
		'''
		Args:
			data:
				Numpy data dict as returned by FlipFlopData.generate_data()

		Returns:
			None.
		'''
		
		super().__init__()
		self.device = get_device(verbose=False)
		self.data = data

	def __len__(self):
		''' Returns the total number of trials contained in the dataset.
		'''
		return self.data['inputs'].shape[0]
	
	def __getitem__(self, idx):
		''' 
		Args:
			idx: slice indices for indexing into the batch dimension of data 
			tensors.

		Returns:
			Dict of indexed torch.tensor objects, with key/value pairs 
			corresponding to those in self.data.

		'''
		
		inputs_bxtxd = torch.tensor(
			self.data['inputs'][idx], 
			device=self.device)

		targets_bxtxd = torch.tensor(
			self.data['targets'][idx], 
			device=self.device)

		return {
			'inputs': inputs_bxtxd, 
			'targets': targets_bxtxd
			}

class FlipFlop(nn.Module):

	def __init__(self, n_inputs, n_hidden, n_outputs, 
		nonlinearity='tanh'):

		super().__init__()

		self.n_inputs = n_inputs
		self.n_hidden = n_hidden
		self.n_outputs = n_outputs
		self.nonlinearity = nonlinearity
		self.device = get_device()

		zeros_1xd = torch.zeros(1, n_hidden, device=self.device)
		self.initial_hiddens_1xd = nn.Parameter(zeros_1xd)
		
		self.rnn = nn.RNN(n_inputs, n_hidden, 
			batch_first=True, 
			nonlinearity=nonlinearity,
			device=self.device)

		# self.rnn = nn.GRU(n_inputs, n_hidden, 
		# 	batch_first=True, 
		# 	device=self.device)

		self.readout = nn.Linear(n_hidden, n_outputs, device=self.device)

		# Create the loss function
		self._loss_fn = nn.MSELoss()
		
	def forward(self, data):
		'''
		Args:
			data: dict of torch.tensor as returned by 
			FlipFlopDataset.__getitem__()

		Returns:
			dict containing the following key/value pairs:
				
				'output': shape (n_batch, n_time, n_bits) torch.tensor 
				containing the outputs of the FlipFlop.

				'hidden': shape (n_batch, n_time, n_hidden) torch.tensor
				containing the hidden unit activitys of the FlipFlop RNN.
		'''

		inputs_bxtxd = data['inputs']
		batch_size = inputs_bxtxd.shape[0]

		# Expand initial hidden state to match batch size. This creates a new 
		# view without actually creating a new copy of it in memory.
		initial_hiddens_1xbxd = self.initial_hiddens_1xd.expand(
			1, batch_size, self.n_hidden)

		# Pass the input through the RNN layer
		hiddens_bxtxd, _ = self.rnn(inputs_bxtxd, initial_hiddens_1xbxd)        

		outputs_bxtxd = self.readout(hiddens_bxtxd)

		return {
			'output': outputs_bxtxd, 
			'hidden': hiddens_bxtxd,
			}

	def predict(self, data):
		''' Runs a forward pass through the model, starting with Numpy data and
		returning Numpy data.

		Args:
			data:
				Numpy data dict as returned by FlipFlopData.generate_data()

		Returns:
			dict matching that returned by forward(), but with all tensors as
			detached numpy arrays on cpu memory.

		'''
		dataset = FlipFlopDataset(data)
		pred_np = self._forward_np(dataset[:len(dataset)])

		return pred_np

	def _tensor2numpy(self, data):

		np_data = {}

		for key, val in data.items():
			np_data[key] = data[key].cpu().numpy()

		return np_data

	def _forward_np(self, data):

		with torch.no_grad():
			pred = self.forward(data)

		pred_np = self._tensor2numpy(pred)

		return pred_np

	def _loss(self, data, pred):

		return self._loss_fn(pred['output'], data['targets'])

	def train(self, train_data, valid_data, 
		learning_rate=1.0,
		batch_size=128,
		min_loss=1e-4, 
		disp_every=1, 
		plot_every=5, 
		max_norm=1.):

		train_dataset = FlipFlopDataset(train_data)
		valid_dataset = FlipFlopDataset(valid_data)

		dataloader = DataLoader(train_dataset, batch_size=batch_size)

		# Create the optimizer
		optimizer = optim.Adam(self.parameters(), 
			lr=learning_rate,
			eps=0.001,
			betas=(0.9, 0.999))

		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer, 
			mode='min',
			factor=.95,
			patience=1,
			cooldown=0)

		epoch = 0
		losses = []
		grad_norms = []
		fig = None
		
		while True:

			t_start = time.time()

			if epoch % plot_every == 0:
				valid_pred = self._forward_np(valid_dataset[0:1])
				fig = FlipFlopData.plot_trials(valid_data, valid_pred, fig=fig)

			avg_loss = 0; avg_norm = 0
			for batch_idx, batch_data in enumerate(dataloader):
				
				# Run the model and compute loss
				batch_pred = self.forward(batch_data)
				loss = self._loss(batch_data, batch_pred)
				
				# Run the backward pass and gradient descent step
				optimizer.zero_grad()
				loss.backward()
				# nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
				optimizer.step()
				
				# Add to the running loss average
				avg_loss += loss/len(dataloader)
				
				# Add to the running gradient norm average
				grad_norms = [p.grad.norm().cpu() for p in self.parameters()]
				grad_norm = np.mean(grad_norms)
				avg_norm += grad_norm/len(dataloader)

			scheduler.step(metrics=avg_loss)
			iter_learning_rate = scheduler.state_dict()['_last_lr'][0]
				
			# Store the loss
			losses.append(avg_loss.item())
			grad_norms.append(avg_norm.item())

			t_epoch = time.time() - t_start
				
			if epoch % disp_every == 0: 
				print('Epoch %d; loss: %.2e; grad norm: %.2e; learning rate: %.2e; time: %.2es' %
					(epoch, losses[-1], grad_norms[-1], iter_learning_rate, t_epoch))

			if loss < min_loss:
				break

			epoch += 1

		valid_pred = self._forward_np(valid_dataset[0:1])
		fig = FlipFlopData.plot_trials(valid_data, valid_pred, fig=fig)

		return losses, grad_norms