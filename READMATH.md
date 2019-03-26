### Variational Autoencoder

---

Variational autoencoders model a latent variable $z$ and an observed variable $x$ via two networks.

1. Encoder network; $q_\phi(z|x)​$ 
2. Decoder network; $p_\theta(x|z)​$

The encoder network tries to approximate the intractable posterior $p_\theta(z|x)$.

The model is trained by maximizing the evidence lower bound via gradient descent.
$$
\mathcal{L}(\theta,\phi) = \mathbb{E}_{z\sim q_\phi(z|x)}[p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p_\theta(z)) \leq p_\theta(x)
$$
We use low-variance Monte Carlo gradients computed via the reparameterization trick [1].

---

In addition, we implement the InfoVAE built on maximum mean discrepancy (MMD) as a choice of divergence [2]. It is built on a modified ELBO,
$$
\mathcal{L}(\theta,\phi) = \mathbb{E}_{z\sim q_\phi(z|x)}[p_\theta(x|z)] - (1-\alpha) D_{KL}(q_\phi(z|x) || p_\theta(z)) -(\alpha+\lambda-1)D_{MMD}(q_\phi(z|x) || p_\theta(z)),
$$

where 
$$
D_{MMD} = \mathbb{E}_{z\sim p,z' \sim p}[k(z,z')] + \mathbb{E}_{z\sim q,z'\sim q}[k(z,z')] - 2\mathbb{E}_{z\sim p, z' \sim q}[k(z,z')],
$$
and we choose the squared exponential kernel
$$
k(z,z') = \exp(-||z-z'||_2^2).
$$


#### Examples

See the `examples/` folder for examples. Below we show the learned latent representation and forward samples for a mixture of Gaussians, with a standard normal prior.

![ex_model](examples/ex_2d.png "Example model output")

#### References

[1] D. P. Kingma & M. Welling, Auto-Encoding Variational Bayes. *International Conference on Learning Representations* (2014).

[2] S. Zhao, J. Song, & S. Ermon, InfoVAE: Information Maximizing Variational Autoencoders. *AAAI Conference on Artificial Intelligence* (2019).


