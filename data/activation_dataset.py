import torch
import numpy as np
import math
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from data.data_utils import ue_prime

"""
the activation_dataset class requires a configuration of the following form:

config = {
    'probe_depth': num,
    'model' : model, #this is the model whose features we want to extract/probe
    'probe_target' : string, #This will be fed into a function which extracts labels based on a predefined function
    'dataset' : dataset, 
    'device' : device, #may want to use a gpu for this.
}

We need to alter the following class. The new name will be 'activation_dataset' and we will use this class both to train probes and to train sparse auto-encoders.
  - The one delicate thing here is the labels. The probe datasets will have non-trivial labels whereas the sparse auto-encoder dataset will not.
  - We know we want to train probes and sparse auto-encoders, but we will likely want additional experiments down the line. What to expect?
"""

class activation_dataset(Dataset):

  #Consists of activations and labels.
  def __init__(self, config):

    self.data_len = len(config['dataset'])
    self.config = config
    self.activations = []
    self.labels = []
    self.device = config['device']

    config['model'].to(self.device)

    for i in tqdm(range(self.data_len)):
      train_acts = self.get_acts(config['model'], config['dataset'], i)
      re, K_r, _ = config['dataset'][i]
      e, r, p = re

      if config['probe_target'] is not None:
        train_labels = get_probe_labels(e, r, p, K_r, config['probe_target'])
        self.labels.append(train_labels)
      else:
        self.labels = None

      self.activations.append(train_acts.detach().cpu())


  def get_acts(self, model, data, idx):
    with torch.no_grad():
      model.eval()
      re, inp, K_r = data[idx]
      act = model(re.unsqueeze(0).to(self.device), inp.unsqueeze(0).to(self.device), self.config['probe_depth'])

    return act


  def __getitem__(self, idx):
    if self.labels:
      return self.activations[idx].squeeze(), self.labels[idx]
    else:
      return self.activations[idx].squeeze()

  def __len__(self):
    return self.data_len




def get_probe_labels(e, r, p, K_r, experiment):
  """
  Collects all the different probing experiments to run.
  """

  with torch.no_grad():

    if experiment == 'pe_prime':
      #counts the number of integers between 1 and re which are divisible by e' but not by p.
      count = 0
      u, e_prime = ue_prime(p,e)
      idx = e_prime.item()

      while idx <= (r*e).item():
        if idx % p != 0:
          count += 1
        idx += e_prime.item()

      return torch.tensor([count]).long()


    if experiment == 'total_summands':
      '''
      Returns the total number of summands in K_{2r-1}(F_{p}[x]/x^e).
      Treat this as a classification problem with r_max*e_max classes.
      The general format for the probe targets allows for multiple dimensions, so 
      we need to unsqueeze.
      '''
      return torch.sum(K_r).unsqueeze(-1)


    if experiment == 'ue_prime':
      """
      Returns the parameters u and e' in Lemma 2 of Speirs' paper.
      Once again, should be a classification problem, but what are the classes? i.e. what is the appropriate range for u and e'?
      u is the maximal exponent of p that is smaller than e - so is bounded above by log_3(40) <4 and e' is less than or equal to 40.
      So label_dims = 2 and label_size = 40 should do the trick.
      """
      u, e_prime = ue_prime(p, e)

      return torch.tensor([u, e_prime]).long()

    if experiment == 're':
      """
      returns the multiplication of r and e.
      Classification task with r_max*e_max classes.
      The general format for the probe targets allows for multiple dimensions, so 
      we need to unsqueeze.
      """
      return r*e.unsqueeze(-1)


    if experiment == 's':
      """
      This experiment attempts to extract the FUNCTION which takes in input p, re, m and spits out the unique integer such that p^{s-1}m <= re < p^s.
      We really think of this as a function of m where p and re are parameters.
      So the label will be of size [r_max*e_max] with indices in between 0 and 8 = ceil(log_3(4000)) and the probe output should be of size [8, r_max*e_max]

      I will be very surprised if we can accurately probe for this, but we need to rule it out before we start looking for other strategies.
      """

      r_max = 100
      e_max = 40
      m = torch.arange(1, r_max*e_max)
      s = torch.where(m > r*e, 0, 1)

      tracker_of_falsehood = True
      exponent = 1

      while tracker_of_falsehood:
        increment = torch.where(r >= (p**exponent)*m, 1, 0)
        if torch.max(increment).item() == 0:
            tracker_of_falsehood = False
        s += increment
        exponent += 1

      return s


    if experiment == 'p_div_count':
      """
      We're going to run through m in the range (1, re) and count how many m are divisible by p.
      """
      count = 0
      for m in range(r*e):
        if m%p == 0:
          count += 1

      return torch.tensor([count]).long() #size (1) or (batch, 1)

    if experiment == 'solution':
      """
      Check to see how accurate each layer is at computing the actual solution.
      """

      return K_r #size (seq_len) or (batch, seq_len)


    if experiment == 'e_prime_div_count':
      """
      We're going to run through m in the range (1, re) and count how many m are both coprime to p and divisible by e'
      """
      #First compute e'
      u, e_prime = ue_prime(p,e)

      #now perform the count
      count = 0

      for m in range(r*e):
        if m%p != 0 and m%e_prime == 0:
          count += 1

      return torch.tensor([count]).long()
