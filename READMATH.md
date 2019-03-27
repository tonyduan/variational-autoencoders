### Variational Autoencoders

---

The VAE models a latent variable $z$ and an observed variable $x$ via two networks.

1. Encoder network; $q_\phi(z|x)​$ 
2. Decoder network; $p_\theta(x|z)​$

The encoder network tries to approximate the intractable posterior $p_\theta(z|x)$.

The model is trained by maximizing the evidence lower bound via gradient descent.
$$
\mathcal{L}(\theta,\phi) = \mathbb{E}_{z\sim q_\phi(z|x)}[p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p_\theta(z)) \leq p_\theta(x)
$$
Low-variance Monte Carlo gradients can be computed via the reparameterization trick.

#### InfoVAE

We implement the InfoVAE [2], built on a modified ELBO,
$$
\mathcal{L}(\theta,\phi) = \mathbb{E}_{z\sim q_\phi(z|x)}[p_\theta(x|z)] - (1-\alpha) D_{KL}(q_\phi(z|x) || p_\theta(z)) -(\alpha+\lambda-1)D(q_\phi(z) || p_\theta(z)),
$$
where the last term is any choice of divergence. In our case we implement two choices: maximum mean discrepancy (MMD) and energy distance [3].
$$
D_{\mathrm{MMD}} = \mathbb{E}_{z\sim p,z' \sim p}[k(z,z')] + \mathbb{E}_{z\sim q,z'\sim q}[k(z,z')] - 2\mathbb{E}_{z\sim p, z' \sim q}[k(z,z')],
$$
$$
D_\mathrm{energy} = 2\mathbb{E}_{z\sim p,z' \sim q}[||z-z'||_2]-\mathbb{E}_{z\sim p,z' \sim p}[||z-z'||_2]-\mathbb{E}_{z\sim q,z' \sim q}[||z-z'||_2].
$$

Our choice of kernel for MMD is the squared exponential kernel.
$$
k(z,z') = \exp(-\frac{||z-z'||_2^2}{\mathrm{dim}}).
$$

#### Examples

See the `examples/` folder for examples. Below we show the learned latent representation and forward samples for a mixture of Gaussians, with a standard normal prior.

![ex_model](examples/ex_2d.png "Example model output")

#### References

[1] D. P. Kingma & M. Welling, Auto-Encoding Variational Bayes. *International Conference on Learning Representations* (2014).

[2] S. Zhao, J. Song, & S. Ermon, InfoVAE: Information Maximizing Variational Autoencoders. *AAAI Conference on Artificial Intelligence* (2019).

[3] M. G. Bellemare, I. Danihelka, W. Dabney, S. Mohamed, B. Lakshminarayanan, S. Hoyer, & R. Munos, The Cramer Distance as a Solution to Biased Wasserstein Gradients (2017).
