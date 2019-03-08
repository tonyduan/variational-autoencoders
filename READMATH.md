### Variational Autoencoder

Variational autoencoders model a latent variable $z$ and an observed variable $x$ via two networks [1].

1. Encoder network; $q_\phi(z|x)​$ 
2. Decoder network; $p_\theta(x|z)​$

The encoder network tries to approximate the intractable posterior $p_\theta(z|x)$.

The model is trained by maximizing the evidence lower bound via gradient descent.
$$
\mathcal{L}(\theta,\phi) = \mathbb{E}_{z\sim q_\phi(z|x)}[p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p_\theta(z)) \leq p_\theta(x)
$$
We use low-variance Monte Carlo gradients computed via the reparameterization trick.

#### Examples

See the `examples/` folder for examples. Below we show the learned latent representation and forward samples for a mixture of Gaussians, with a standard normal prior.

![ex_model](examples/ex_2d.png "Example model output")

#### References

[1] D. P. Kingma & M. Welling, Auto-Encoding Variational Bayes. *Proceedings of the 2nd International Conference on Learning Representations* (2014).