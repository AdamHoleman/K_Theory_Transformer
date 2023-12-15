import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import copy


from data.data_generation import trun_poly_dataset
from models.modules import probeable_decoder_model
from data.data_utils import ue_prime
from data.activation_dataset, import get_probe_labels





#-------------------------- GENERAL UTILITIES -----------------------------


def get_learned_feats(model, encoder, re, K_r, device):
  #given input, return the encoded activations

  input = torch.cat((torch.zeros(1), K_r), dim = 0)[0:K_r.size(0)].long() #append start token
  act = model(re.unsqueeze(0).to(device), input.unsqueeze(0).to(device), 2)

  return F.relu(encoder.encoder(act)) #remember to apply ReLU! (should be of size (1, seq_len, latent_dim))



def inspect_data(datapoints):
  #given a collection of datapoints, print out a collection of facts about this dataset.

  p_vals = []
  r_vals = []
  e_vals = []
  u_vals = []
  e_prime_vals = []

  pe_prime_counts = []

  p_div_counts = []


  for re in tqdm(datapoints):

    #The datapoints
    e_vals.append(re[0].item())
    r_vals.append(re[1].item())
    p_vals.append(re[2].item())

    #u and e' features
    u, e_prime = ue_prime(re[2], re[0])
    u_vals.append(u.item())
    e_prime_vals.append(e_prime.item())

    #def get_feature(e, r, p, K_r, feature):

    count = get_probe_labels(re[0], re[1], re[2], 0, 'pe_prime').item()
    pe_prime_counts.append(count)

    count = get_feature(re[0], re[1], re[2], 0, 'p_div_count').item()
    p_div_counts.append(count


  print(" ")
  print(f'p_vals: {set(p_vals)}')
  print(f'r_vals: {set(r_vals)}')
  print(f'e_vals: {set(e_vals)}')
  print(f'e_prime_vals: {set(e_prime_vals)}')
  print(f'u_vals: {set(u_vals)}')
  print(f"pe_prime divisibility count = {set(pe_prime_counts)}")
  print(f"p_div_count = {set(p_div_counts)}")






#------------------------------- EXPERIMENT 1 -----------------------------------

def measure_densities(data, latent_dim):
  #runs through the data and returns a tensor of size latent_dim whose ith index is the density of the ith feature.

  densities = torch.zeros(latent_dim)

  for re, K_r in tqdm(data):

    learned_feats = get_learned_feats(model, encoder, re, K_r)

    active_neurons = torch.where(learned_feats == 0, 0, 1)
    
    #SHOULD LOOK AT THOSE NERUONS WHICH ACTIVATE AT THE FIRST TOKEN
    densities += active_neurons.squeeze()[0]

  return densities/(len(data))


def activating_data(data, idx, position = 1):
  #given the index of a neuron and a position, collect all datapoints which activate the neuron at that location.
  #default is 1, as this position exhibits the most specialization
  datapoints = []

  for re, K_r in tqdm(data):
    learned_feats = get_learned_feats(model, encoder, re, K_r)

    #restrict to the specified position
    learned_feats = learned_feats.squeeze()[position]

    if learned_feats[idx] != 0:
      datapoints.append(re)

  return datapoints



#--------------------------- EXPERIMENT 3 ----------------------------

def positional_activations(data, encoder, model):
  device = 'cuda'
  tot_count = torch.zeros(encoder.latent_dim).to(device)
  pos_count = torch.zeros(data.seq_len, encoder.latent_dim).to(device)

  for re, K_r in tqdm(data):
    learned_feats = get_learned_feats(model, encoder, re, K_r, device = device).squeeze()

    acts = torch.where(learned_feats > 1e-10, 1, 0)
    pos_count += acts
    tot_count += torch.sum(acts, dim=0)

  return pos_count/tot_count



def visualize_neuron_summands(model, encoder, neurons, activating_data, position=1):
  #given a list of neurons and a list of the datapoints which activate the neurons at the specified position, plot activation level vs. summand count.
  #for each neuron, we will count the number of z/p^{position+1} summands in data which activates the each neuron
  #In addition, we will record the activation levels of the neuron at each datapoint.

  neuron_summands = [[] for i in range(len(neurons))] #for histogram
  pairs = [[] for i in range(len(neurons))] #for scatterplot

  for i, idx in enumerate(neurons):
    for re, K_r in activating_data[i]:
      neuron_summands[i].append(K_r[position].item())

      learned_feats = get_learned_feats(model, encoder, re, K_r).squeeze()
      lf = learned_feats[position]
      activation = lf[idx].item()
      pairs[i].append((activation, K_r[position].item()))

  for i, idx in enumerate(neurons):
    print(" ")

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
    fig.suptitle(f'Z/p^{position +1} summands for neuron {idx}')

    ax1.hist(neuron_summands[i], bins=50, alpha = 0.7, label=f'{position}-summands where {idx} activates')
    ax2.scatter(*zip(*pairs[i]), alpha = 0.7)
    ax1.legend()

    plt.show()




