import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


#STILL NEED NEURON RESAMPLING!

class Encoder_Trainer:
	#the datasets should consist of the activations of the base model. The model for the trainer is the sparse autoencoder.
	def __init__(self, model, train_dataset, test_dataset, config):
		self.model = model
		self.train_dataset = train_dataset
		self.test_dataset = test_dataset
		self.config = config
		self.iter = 0

		# take over whatever gpus are on the system
		self.device = 'cpu'
		if torch.cuda.is_available():
			self.device = torch.cuda.current_device()

		#move the model to the device
		self.model.to(self.device)

		# log something for plotting
		self.train_loss_cont = []
		self.rec_loss_cont = []
		self.sparsity_loss_cont = []

		self.test_loss_cont = []


	def plot_logdensity(self, neuron_fires):
		#plots a histogram of the log-densities of the latent variable
		log_density = np.log10(neuron_fires.cpu().numpy()/(8*len(self.train_dataset)) + 1e-10)

		N, bins, patches = plt.hist(log_density, bins=60, range = (max(-7, min(log_density)),max(log_density)), weights = np.ones(len(log_density))/len(log_density))
		fracs = N/N.max()
		# we need to normalize the data to 0..1 for the full range of the colormap
		norm = colors.Normalize(fracs.min(), fracs.max())

		# Now, we'll loop through our objects and set the color of each accordingly
		for thisfrac, thispatch in zip(fracs, patches):
			color = plt.cm.viridis(norm(thisfrac))
			thispatch.set_facecolor(color)

		plt.show()



	def train(self, printing=True):
		model, config = self.model, self.config

		optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'])


		def run_one_epoch(split):

			is_train = split == 'train'
			model.train(is_train)
			data = self.train_dataset if is_train else self.test_dataset

			loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config['batch_size'],
                                num_workers=1)


			neuron_fires = torch.zeros(model.latent_dim, requires_grad=False).to(self.device) #stores the number of times a given neuron fires

			losses = []
			rec_losses = []
			sparsity_losses = []

			pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)

			for it, act in pbar:
				self.iter += 1

				act = act.to(self.device)  #should be size (batch, seq_len, dim_feedforward)

				with torch.set_grad_enabled(is_train):

					#forward the model
					loss, rec_error, sparsity_error, rec, f = model(act) #forward the model
					losses.append(loss.item())
					rec_losses.append(rec_error.item())
					sparsity_losses.append(sparsity_error.item())

					#track neuron sparsity
					fired_neurons = torch.where(torch.abs(f.detach()) > 1e-10, 1, 0) #should be of size (batch, seq_len, latent_dim)

					neuron_fires += torch.sum(fired_neurons, dim = (0,1)) #so this records the total number of fires on all tokens in a batch.
					num_dead_neurons = torch.count_nonzero(neuron_fires == 0)


				if is_train:
					#perform backward pass
					model.zero_grad()
					loss.backward()

					#must update the gradients of the decoder before stepping the optimizer!
					model.project_grads()

					#clip gradients and step the optimizer
					torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm_clip'])
					optimizer.step()

					#must renormalize the decoder columns before forwarding the model!
					model.normalize_decoder_cols()

					mean_loss = float(np.mean(losses))
					mean_rec_loss = float(np.mean(rec_losses))
					mean_sparsity_loss = float(np.mean(sparsity_losses))

					pbar.set_description(f"epoch {epoch+1}: mean reconstruction loss {mean_rec_loss:.5f}, mean sparsity loss {mean_sparsity_loss:.5f}, number of dead neurons = {num_dead_neurons}")



          #--------------------------- ~~~ TRACK STATS AND PLOT ~~~ -------------------------------
			if is_train:
				self.train_loss_cont.append(mean_loss)


			if not is_train:
				test_loss = float(np.mean(rec_losses))

				if printing:
					self.plot_logdensity(neuron_fires)
					print(f"test reconstruction loss {test_loss:.5f}")
					print("")
					self.test_loss_cont.append(test_loss)

				return test_loss



		best_loss = float('inf')

		for epoch in range(config['epochs']):

			if epoch == 5:
				torch.save(self.model.state_dict(), 'epoch5_encoder_state_dict')

			run_one_epoch('train')
			if self.test_dataset is not None:
				test_loss = run_one_epoch('test')
				if test_loss < best_loss:
					best_loss = test_loss

		#torch.save(encoder.state_dict(), 'epoch10_encoder_state_dict')





