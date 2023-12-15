import torch
import numpy
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

import data.data_generation
from data.data_generation import trun_poly_dataset
from models.K_transformer import probeable_decoder_model
from data.data_utils import accuracy





class model_trainer():

	def __init__(self, model, config):
		self.model = model
		self.device = config['device']
		self.epochs = config['epochs']
		self.optim = self.configure_optimizer(model, config['lr'])
		self.train_iter = self.configure_dataloader(config['trainset'], config['batch_size'], config['device'])
		self.val_iter = self.configure_dataloader(config['valset'], config['batch_size'], config['device'])

		self.training_examples = len(config['trainset'])


	def configure_optimizer(self, model, lr):
		return torch.optim.AdamW(model.parameters(), lr = lr)

	def configure_dataloader(self, dataset, batch_size, device):
		device = self.device
		return torch.utils.data.DataLoader(dataset, 
												batch_size = 512, 
												collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)), 
												shuffle = True)

	def train(self, save_progress=False):
		device = self.device
		model = self.model
		epochs = self.epochs
		optim = self.optim

		train_iterator = self.train_iter
		val_iterator = self.val_iter
		val_acc_list = []
		train_loss = []
		min_acc = .2 #initialize bound for saving best performing model based on validation accuracy.

		for i in range(epochs):
			#prep trackers for training

			model.train()
			tot_loss = 0
			for re, K_r, labels in train_iterator:
				optim.zero_grad()
				model = model.to(device)

				pred = model(re, K_r)  #size (batch, seq_len, num_classes)
				pred = torch.permute(pred, (0,2,1)) #prep for cross entropy loss

				loss = F.cross_entropy(pred, labels, reduction = 'mean')
				loss.backward()
				optim.step()
				tot_loss += loss.detach().item()/self.training_examples

			train_loss.append(tot_loss)

			if i%10 == 0:
				print("Epoch {} training loss = {:.6f}".format(i+1, tot_loss))
				print("Epoch {} training accuracy = {:.6f}".format(i+1, accuracy(model, train_iterator)))

		#prep trackers for validation
		model.eval()
		val_acc = accuracy(model, val_iterator)

		if save_progress:
			if (i+1)%10 == 0:
				torch.save(model.state_dict(), f'{model_path}_epoch{i+1}')

			if val_acc >= min_acc:
				min_acc = val_acc
				torch.save(model.state_dict(), "best_performing_model")


		val_acc_list.append(val_acc)

		print("Epoch {} validation accuracy = {:.6f}".format(i+1, val_acc))

		return train_loss, val_acc_list