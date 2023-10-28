import torch
import numpy

def ue_prime(p, e):
    """
    e is an integer
    p is a prime
    returns the unique integers (dtype = torch.int64) u, e' such that (e',p) = 1 and e = p^{u}e'.
    """

    divisible = True
    u = 0
    e_prime = 1

    while divisible:
        q = p**u
        if e % q == 0:
            e_prime = e/q
            u += 1
        else:
            divisible = False
            u += -1

    return torch.tensor([u]), torch.tensor([e_prime])


def get_s(p, r, d):
    """
    p is a prime, both r and d are integers
    d is an integer (or batch of integers)
    returns the unique positive (batch of) integer(s) (dtype = torch.int64) s such that
        p^{s-1}d <= r < p^{s}d
    if it exists, and zero otherwise.
    """
    init = torch.where(d > r, 0, 1) #all zero indices are correctly specified at this point
    s = init

    tracker_of_falsehood = True
    exponent = 1

    while tracker_of_falsehood:
        increment = torch.where(r >= (p**exponent)*d, 1, 0)
        if torch.max(increment).item() == 0:
            tracker_of_falsehood = False
        s += increment
        exponent += 1

    return s


def get_h(p, r, e, m_prime):

    u, e_prime = ue_prime(p, e)
    s = get_s(p,r*e, m_prime)
    out = torch.where(m_prime % e_prime == 0, torch.minimum(s, u), s)

    return out


def get_indices(p, r, e):
    """
    returns the list of all integers m (dtype = torch.int64) such that 1 <= m <= re and (m,p) = 1
    """

    indices = []

    for m in range(1, r*e + 1):
        if m % p != 0:
            indices.append(torch.tensor([m]))

    return torch.cat(indices)



def encoding_scheme(tor_list, seq_len):
    """
    Input: list of integers [h_{1},..., h_{n}] between 1 and seq_len
    Output: a tensor [x_{1},..., x_{m}] where x_{i} is the number of occurences of i in the input
    """

    seq = torch.zeros(seq_len)

    for i in range(1, seq_len):
        seq[i-1] = torch.sum(torch.where(tor_list == i, 1, 0))

    return seq



def prepare_input(K_r):
  #Adds a start token to target sequence to prep for auto-regressive transformer
  device = K_r.device
  batch_size = K_r.size(0)
  return torch.cat((torch.zeros(batch_size, 1).to(device), K_r), dim = 1)[:, 0:K_r.size(1)].long()


def accuracy(model, iterator):
  error_counter = 0
  size = 0
  for re, K_r in iterator:
    size += re.size(0) #keep track of batch sizes
    input = prepare_input(K_r)

    pred = model(re, input)

    error = (torch.argmax(pred, dim=2) - K_r).cpu().numpy()
    errors = numpy.any(error, axis = 1)
    error_counter += numpy.sum(errors.astype(int))

  return 1- error_counter/size