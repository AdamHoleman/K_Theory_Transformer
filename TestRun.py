import torch

from data.data_generation import trun_poly_dataset
from models.K_transformer import probeable_decoder_model
from data.activation_dataset import activation_dataset
from models.probe_trainer import Trainer as probe_trainer
from models.sparse_autoencoder import sparse_autoencoder
from models.model_trainer import model_trainer
from models.linear_probe import linear_probe
from models.autoencoder_trainer import Encoder_Trainer



"""
This script basically runs through the whole project to make sure all the code works
	- generates a tiny dataset (1000 examples), 
	- trains a small model on it, 
	- computes the accuracy, 
	- generates an activation dataset
	- runs a probing experiment
	- train a sparse autoencoder to extract features.
"""



if __name__ == '__main__':

	"""
	generate a dataset with 10 primes, 10 e values and 10 r values
	"""
	primes = [2,3,5,7,11, 13, 17, 19, 23, 29]
	dataset = trun_poly_dataset(primes, 10, 10)

	train_data, val_data = torch.utils.data.random_split(dataset, [800, 200])


	"""
	setup a tiny model
	"""
	n_heads = 4
	emb_dim = 32
	dim_feedforward = 128
	num_layers = 2
	vocab_size = 5000
	seq_len = dataset.seq_len

	model = probeable_decoder_model(emb_dim, n_heads, vocab_size, seq_len, num_layers, dim_feedforward)



	"""
	setup a training configuration and train the model
	"""
	train_config = {
					'device' : 'cpu',
					'epochs' : 20,
					'lr' : .01,
					'trainset': train_data,
					'valset' : val_data,
					'batch_size' : 4
	}

	trainer = model_trainer(model, train_config)
	trainer.train()


	"""
	let's build an activation dataset at the mlp layer and initialize a probe
	"""

	probe_config = {
    				'probe_depth': 2,
    				'model' : model,
    				'probe_target' : 'p_div_count', #This will be fed into a function which extracts labels based on a predefined function
    				'target_size' : vocab_size,
    				'target_dims' : 1,
    				'dataset' : dataset,
    				'device': 'cpu',
					}

	probe_training_config = {
    				'grad_norm_clip' : 10.0,
    				'lr' : 1e-5,
    				'epochs' : 5,
    				'batch_size' : 32,
					}


	probe = linear_probe(dim_feedforward, dataset.seq_len, probe_config['target_size'], probe_config['target_dims'])
	act_data = activation_dataset(probe_config)

	act_train, act_val = torch.utils.data.random_split(act_data, [800, 200])

	act, label = act_data[0] 


	"""
	Set up a trainer and train the probe
	"""
	probe_trainer = probe_trainer(probe, act_train, act_val, probe_training_config)
	probe_trainer.train()


	"""
	Finally, train a sparse auto-encoder. Unfortunately, I've set up the model_trainer to not account for labels, so we need a new dataset. Oh well!
	"""
	encoder_data_config = {
    				'probe_depth': 2,
    				'model' : model,
    				'probe_target' : None, #This will be fed into a function which extracts labels based on a predefined function
    				'target_size' : vocab_size,
    				'target_dims' : 1,
    				'dataset' : dataset,
    				'device': 'cpu',
					}

	encoder_training_config = {
    				'grad_norm_clip' : 10.0,
    				'lr' : 1e-5,
    				'epochs' : 5,
    				'batch_size' : 32,
					}

	act_data = activation_dataset(encoder_data_config)
	enc_train, enc_val = torch.utils.data.random_split(act_data, [800, 200])


	encoder = sparse_autoencoder(dim_feedforward, 1024, .001)
	encoder_trainer = Encoder_Trainer(encoder, enc_train, enc_val, encoder_training_config)
	encoder_trainer.train()



























