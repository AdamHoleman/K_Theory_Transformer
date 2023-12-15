from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from data.data_generation import trun_poly_dataset
import copy



class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

        #move the model to the device
        self.model.to(self.device)

        # log something for plotting
        self.train_loss_cont = []
        self.test_loss_cont = []
        self.train_acc_cont = []
        self.test_acc_cont = []


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

          losses = []

          pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)

          for it, (act, y) in pbar:
              act = act.to(self.device)  #should be size (batch, seq_len, emb_dim) or (batch, seq_len, dim_feedforward) depending on probe depth.
              y = y.to(self.device)      #varies depending on task - usually a batch of class labels

              with torch.set_grad_enabled(is_train):
                  logits, loss = model(act, y)
                  loss = loss.mean()
                  losses.append(loss.item())

              if is_train:
                  # backprop and update the parameters
                  model.zero_grad()
                  loss.backward()
                  torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm_clip'])
                  optimizer.step()
                  mean_loss = float(np.mean(losses))
                  pbar.set_description(f"epoch {epoch+1}: train loss {mean_loss:.5f}")

          if is_train:
              self.train_loss_cont.append(mean_loss)

          if not is_train:
              test_loss = float(np.mean(losses))
              test_acc = self.get_accuracy(loader)

              if printing:
                  print(f"test loss {test_loss:.5f}; test acc {test_acc*100:.2f}%")
                  print("")
                  self.test_loss_cont.append(test_loss)
                  self.test_acc_cont.append(test_acc)

              return test_loss

        best_loss = float('inf')

        for epoch in range(config['epochs']):
            run_one_epoch('train')
            if self.test_dataset is not None:
              test_loss = run_one_epoch('test')
              if test_loss < best_loss:
                best_loss = test_loss
                #self.save_checkpoint() - TO DO: Add in save_checkpoint function.



    def get_accuracy(self, dataloader):
      error_counter = 0

      with torch.no_grad():
        for acts, labels in dataloader:
          #Labels will be of shape (batch, num_classes, label_dims)
          logits = self.model(acts.to(self.device)) 
          preds = torch.argmax(logits, dim=1).cpu() #will be size [batch, label_dims]

          errors = (preds - labels)
          errors = torch.where(errors == 0, 0, 1) #still of size [batch, label_dims], but all non-zero values have been replaced with 1
          errors = torch.where(torch.sum(errors,dim=1)==0, 0, 1)
          error_counter += torch.sum(errors).item()

      return 1 - error_counter/len(dataloader.dataset)






