# K Theory Transformer

This is part of an ongoing project to apply machine learning to the computation of algebraic K-theory.

<center> <h2> What is algebraic K-theory?</h2> </center>

Algebraic $K$-theory is a mathematical invariant of rings, algebraic varieties, or more general mathematical objects depending on the context. Given a ring $R$, algebraic $K$-theory assigns to $R$ a sequence of abelian groups
   $$K_{r}(R), r \in \mathbb{N}$$
which captures deep arithmetic and geometric information regarding the ring $R$.

The modern version of algebraic K-theory was introduced by Daniel Quillen in 1971, for which he was awarded the Fields medal. Since its introduction, algebraic K-theory has become a central topic of investigation within the mathematical community, playing a crucial role in many open conjectures that connect widely disparate fields of math. Despite its importance, the actual computation of algebraic K-theory remains highly mysterious. For example, we still don't know how to describe the algebraic K-theory of the integers, $\mathbb{Z}$.

Classical results of Hesselholt and Madsen as well as recent advances of Antieau, Krause, and Nikolaus have made possible the collection of a large dataset describing the algebraic $K$-theory of a certain class of rings. The goal of this project is to bring to bear the techniques of deep learning to this newly available data and hopefully extract new insights in to the nature of the this enigmatic mathematical invariant.

<center> <h2> Data Collection </h2> </center>

For this stage of the project, we focus our attention on the class of truncated polynomial rings over finite fields:
  $$R = \mathbb{F}_{p}[x]/(x^e)$$
where $p$ is a prime number. In this setting, a classical result of Hesselholt and Madsen allows us to write down the $K$-groups of $R$ by appling a simple algorithm:

<h3> Algorithm 1:</h3> 
<ul>
<li> <b>Input:</b> a prime number $p$, an integer $e$, and an integer $r$ </li>
<li> <b>Output:</b> a list of integers $L = [h_1, ..., h_{l}]$ such that

  $$K_{2r-1}(R, (x)) \simeq \bigoplus_{i=1}^{l} \mathbb{Z}/p^{h_{i}}\mathbb{Z}$$</li>
</ul>
<ol>
  <li> Find the unique pair of integers $(u, e')$ such that $e'$ is not divisible by $p$ and $e = p^{u}e'$. </li>
  <li> Initialize an empty list $L$ and loop through all integers $m$ in the range $(1, re)$, performing the following steps:
      <ul>
        <li> If $m$ is divisible by $p$, move to the next step of the loop. </li>
        <li> Find the unique integer $s$ such that $p^{s-1} m \leq re < p^{s}m$. </li>
        <li> If $m$ is  divisible by $e'$, add the minimum of $\{u, s\}$ to $L$.</li>
        <li> If $m$ is not divisible by $e'$, add $s$ to $L$. </li>
      </ul>  
  <li> Return the list $L$. </li>
</ol>


Understanding why this algorithm works requires significant mathematical background (for the adventurous, this is equivalent to combining Theorem 1 and Lemma 2 in Martin Speirs paper 'On the K-Theory of Truncated Polynomial Rings Revisited'). On the other hand, if we accept the validity of this algorithm, it gives us a means to produce a sequential dataset describing the $K$-groups of truncated polynomial rings. In particular, we could train a transformer/RNN/LSTM to auto-regressively predict the algebraic $K$-groups of truncated polynomial rings.

<h3> Computational Consideration</h3>
We now describe some of the considerations that go in to using this algorithm to produce a dataset which allows us to efficiently and accurately train an autoregressive model.

<h4> Decreasing the Sequence Length in Exchange for Larger Vocabulary</h4>

Note that in the algorithm above, we loop through all integers $m \in (1, re)$ and for each $m$ which is not divisible by $p$, we add an item to out list. The result is a list that is of length $re - \lfloor \frac{re}{p}\rfloor$. Since $r$ and $e$ will vary throughout the dataset, we will have sequences of varying sizes which we will pad to the maximal sequence length, which is given by
  $$\text{naive max sequence length} = r_{\text{max}} \cdot e_{\text{max}} - \lfloor \frac{r_{\text{max}} \cdot e_{\text{max}}}{p} \rfloor$$
How reasonable is it to take this as our sequence length? Well, the dataset that I use during training has two million examples, with $r_{\text{max}} = 100$ and $e_{\text{max}} = 50$, so with this formulation our sequence length would be approximately 5000 tokens long! This is an extremely large context length for a transformer - especially a small one which we might hope to thoroughly explore with interpretability techniques.

On the other hand, our vocabulary size (i.e. the number of possible integers that could show up in the sequence) is equal to the base $p$ logarithm of $r_{\text{max}} \cdot e_{\text{max}}$. Since $p$ varies, we will bound this by taking the smallest $p$ in our dataset which is $p = 2$. This gives us
  $$\text{naive vocabulary size} = \lceil log_{2}(r_{\text{max}} \cdot e_{\text{max}}) \rceil$$
which is extremely small!

The key insight here is that we can relabel our data by counting the number of words that show up in the algorithm. So our labelling procedure can be reformulated as:
<ul> 
<li> Input: triples of integers $(p, r, e)$</li>
<li> Label: list $L' = [n_1, n_2,..., n_k]$ where $n_i$ is the number of occurrences of $i$ in the list $L$ obtained from Algorithm 1. </li>
</ul>

With these new labels we effectively exchange the vocabulary size and sequence length, yielding a dataset which is far more amenable to autoregressive techniques.
$$\text{new max sequence length} = \lceil log_{2}(r_{\text{max}} \cdot e_{\text{max}}) \rceil$$
$$\text{new vocabulary size} = r_{\text{max}} \cdot e_{\text{max}} - \lfloor \frac{r_{\text{max}} \cdot e_{\text{max}}}{p} \rfloor$$

<h4> Varying $p$ as We Scale The Dataset</h4>

In order to create a dataset, we need only specify a collection of prime numbers $\{p\}$, and collections of integers $\{r\}, \{e\}$. We then apply Algorithm 1 to each triple $(p,r,e)$ and reformat the labels as described above. The resulting size of the dataset $D$ is obtained by multiplying the sizes of these three sets:
  $$\text{len}(D) =  \text{number of primes} \cdot \text{number of }r \cdot \text{number of }e$$
If we fix the size of dataset (say, at two million examples), there are thus many different ways to generate these examples. For example, we could fix a single prime $p$, vary $r$ between 1 and 2000, and vary $e$ between 1 and 1000. 

Notice that the choice of these sets also determines the vocabulary size and sequence length. So in the above example, we would produce a dataset with vocabulary size given by $2000000 - \lfloor 2000000/p \rfloor$, and a sequence length of $log_{p}(2000000)$. While the sequence length is fairly reasonable, the vocabulary size is massive (more than a million words)!

Alternatively, we can achieve the same size of dataset with far smaller vocabulary size and sequence length by varying the primes $p$. We will take our set of primes to consist of the first 400 primes, allow $r$ to range between $1$ and $100$, and allow $e$ to range between $1$ and $50$. This yields a dataset such that
  $$\text{sequence length} = \lceil log_{2}(5000) \rceil = 13$$
  $$\text{vocabulary size} \leq 5000$$
This is a massive improvement over a more naive implementation and highlights the necessity of varying the prime $p$.



<h2>Model Architecture and Training</h2>

<h2> Probing Experiments</h2>


