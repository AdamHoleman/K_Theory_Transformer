import torch
from data.data_utils import ue_prime, get_s, get_h, get_indices, encoding_scheme
import math
from tqdm import tqdm


class trun_poly_dataset(torch.utils.data.Dataset):
    
    def __init__(self, primes, max_e, max_r):
        """
        CAUTION: we want to make sure max_e + max_r < self.seq_len for the following to compile - this should hold for sufficiently large e and r.
        """     
        self.num_examples = max_e*max_r*len(primes)      
        self.max_generators = max_e*max_r - (max_e*max_r)//max(primes) #this will give us an estimate of the vocab size
        self.seq_len = math.ceil(math.log(max_e*max_r, min(primes)))+1 #this will give us the sequence length
        
        
        data = torch.zeros(self.num_examples, 3)
        labels = torch.zeros(self.num_examples, self.seq_len)
        counter = 0
        prime_counter = 0
        
        for p in tqdm(primes):
            prime_counter += 1
            for e in range(1, max_e+1): 
                for r in range(1, max_r +1):
                    #INPUT VALUES - encode e, r, and p in the first three entries.
                               
                    data[counter] = torch.tensor([e,r,p])
                
                    #LABELS
                    m_primes = get_indices(p, r, e)
                    h = get_h(p,r,e,m_primes)

                    labels[counter] = encoding_scheme(h, self.seq_len)
                    counter += 1
                
        self.data = data
        self.labels = labels

    def prepare_input(self, K_r):
        #Adds a start token to target sequence to prep for auto-regressive transformer
        return torch.cat((torch.zeros(1), K_r), dim = 0)[0:K_r.size(0)].long()
        
    def __len__(self):
        return self.num_examples
        
    def __getitem__(self, idx):

        K_r = self.labels[idx]  
        inp = self.prepare_input(K_r) 

        return self.data[idx].long(), inp.long(), K_r.long()











