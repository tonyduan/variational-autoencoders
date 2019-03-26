### Variational Autoencoder

---

Variational autoencoders model a latent variable <img alt="$z$" src="svgs/f93ce33e511096ed626b4719d50f17d2.svg" align="middle" width="8.367621899999993pt" height="14.15524440000002pt"/> and an observed variable <img alt="$x$" src="svgs/332cc365a4987aacce0ead01b8bdcc0b.svg" align="middle" width="9.39498779999999pt" height="14.15524440000002pt"/> via two networks.

1. Encoder network; <img alt="$q_\phi(z|x)​$" src="svgs/8c4290cd764b7be62885f1f2fa0f1ace.svg" align="middle" width="51.17860604999999pt" height="24.65753399999998pt"/> 
2. Decoder network; <img alt="$p_\theta(x|z)​$" src="svgs/7b4b76719fd0c5230e3c2d4849ba0924.svg" align="middle" width="50.82199814999999pt" height="24.65753399999998pt"/>

The encoder network tries to approximate the intractable posterior <img alt="$p_\theta(z|x)$" src="svgs/8d064232b1495aa703d8d2bb1a19d3aa.svg" align="middle" width="50.82199814999999pt" height="24.65753399999998pt"/>.

The model is trained by maximizing the evidence lower bound via gradient descent.
<p align="center"><img alt="$$&#10;\mathcal{L}(\theta,\phi) = \mathbb{E}_{z\sim q_\phi(z|x)}[p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p_\theta(z)) \leq p_\theta(x)&#10;$$" src="svgs/2e6a68c5089ed72134a0cf0384848df5.svg" align="middle" width="427.3819407pt" height="18.639307499999997pt"/></p>
We use low-variance Monte Carlo gradients computed via the reparameterization trick [1].

---

In addition, we implement the InfoVAE built on maximum mean discrepancy (MMD) as a choice of divergence [2]. It is built on a modified ELBO,
<p align="center"><img alt="$$&#10;\mathcal{L}(\theta,\phi) = \mathbb{E}_{z\sim q_\phi(z|x)}[p_\theta(x|z)] - (1-\alpha) D_{KL}(q_\phi(z|x) || p_\theta(z)) -(\alpha+\lambda-1)D_{MMD}(q_\phi(z|x) || p_\theta(z)),&#10;$$" src="svgs/fec0fe6699fc8ed2cd2f53cd80f069e0.svg" align="middle" width="688.2882434999999pt" height="18.639307499999997pt"/></p>

where 
<p align="center"><img alt="$$&#10;D_{MMD} = \mathbb{E}_{z\sim p,z' \sim p}[k(z,z')] + \mathbb{E}_{z\sim q,z'\sim q}[k(z,z')] - 2\mathbb{E}_{z\sim p, z' \sim q}[k(z,z')],&#10;$$" src="svgs/c53d14625258393709a1c37edb458088.svg" align="middle" width="509.09157749999997pt" height="17.8831554pt"/></p>
and we choose the squared exponential kernel
<p align="center"><img alt="$$&#10;k(z,z') = \exp(-||z-z'||_2^2).&#10;$$" src="svgs/e7492173287d558b5f473bc7d9477f3f.svg" align="middle" width="194.76039165pt" height="18.312383099999998pt"/></p>


#### Examples

See the `examples/` folder for examples. Below we show the learned latent representation and forward samples for a mixture of Gaussians, with a standard normal prior.

![ex_model](examples/ex_2d.png "Example model output")

#### References

[1] D. P. Kingma & M. Welling, Auto-Encoding Variational Bayes. *International Conference on Learning Representations* (2014).

[2] S. Zhao, J. Song, & S. Ermon, InfoVAE: Information Maximizing Variational Autoencoders. *AAAI Conference on Artificial Intelligence* (2019).


