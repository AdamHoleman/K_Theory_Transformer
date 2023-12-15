import torch
import torch.nn as nn
import torch.nn.functional as F



class sparse_autoencoder(nn.Module):
    def __init__(self, activation_dim, latent_dim, sparsity):
      super().__init__()

      self.latent_dim = latent_dim

      #layers
      self.encoder = nn.Linear(activation_dim, latent_dim)
      self.decoder = nn.Linear(latent_dim, activation_dim, bias = False) #we want to do something funky with the decoder bias, so we'll customize.

      #biases
      self.dec_bias = nn.Parameter(torch.zeros(activation_dim), requires_grad = True)

      #hyperparameters
      self.l = sparsity

      #initialize weights()
      nn.init.kaiming_uniform_(self.encoder.weight)
      nn.init.kaiming_uniform_(self.decoder.weight)
      self.normalize_decoder_cols()


    def forward(self, act):
      #returns the training loss.
      act = act - self.dec_bias
      f = F.relu(self.encoder(act))
      rec = self.decoder(f) + self.dec_bias

      rec_error = F.mse_loss(rec, act)
      sparsity_error = self.l*F.l1_loss(f, torch.zeros_like(f), reduction = 'sum')

      #return all data we need for downstream tasks: loss, two errors, the reconstruction, and the featurses

      return rec_error + sparsity_error, rec_error, sparsity_error, rec, f


    @torch.no_grad()
    def project_grads(self):
      #Since we're constraining the decoder's columns to the unit sphere, the gradients must be constrained to the tangent space of the sphere
      #i.e. the orthogonal complement of the columns themselves.

      if self.decoder.weight.grad is not None:
        proj_W_grad = (self.decoder.weight.grad*self.decoder.weight).sum(-1, keepdim=True) * self.decoder.weight
        self.decoder.weight.grad = self.decoder.weight.grad - proj_W_grad


    @torch.no_grad()
    def normalize_decoder_cols(self):
      #normalizes the columns of the decoder and updates the gradient.
      #Note that the manifold of unit length vectors in R^{n} is merely the n-1 sphere S^{n-1}, so the tangent space at a vector v in S^{n-1}
      #can be naturally identified with the orthogonal complement of v in R^{n}. Hence, it's both necessary and sufficient for the columns of
      #the decoder's gradient to be orthogonal to the columns of the weights
      self.project_grads()

      #normalize the weights
      norm = torch.norm(self.decoder.weight, dim = -1, keepdim= True)
      norm = torch.clamp(norm, 1e-8)
      normalized_dec_weights = self.decoder.weight/norm

      self.decoder.weight.data = normalized_dec_weights