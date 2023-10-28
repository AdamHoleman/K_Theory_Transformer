import torch
from data.data_utils import ue_prime, get_s, get_h, get_indices, encoding_scheme


class trun_poly_dataset(torch.utils.data.Dataset):
    
    def __init__(self, primes, max_e, max_r):
        """
        CAUTION: we want to make sure max_e + max_r < self.seq_len for the following to compile - this should hold for sufficiently large e and r.
        """     
        self.num_examples = max_e*max_r*len(primes)      
        self.max_generators = self.num_examples - self.num_examples//max(primes) #this will give us the vocab size
        self.seq_len = math.ceil(math.log(max_e*max_r, min(primes)))+1 #this will give us the sequence length
        
        
        data = torch.zeros(self.num_examples, 3)
        labels = torch.zeros(self.num_examples, self.seq_len)
        counter = 0
        prime_counter = 0
        
        for p in primes:
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
                
            if e%10 == 0:
                print('K groups of the first {} rings computed!'.format(prime_counter * e))
                
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return self.num_examples
        
    def __getitem__(self, idx):         
        return self.data[idx].long(), self.labels[idx].long()