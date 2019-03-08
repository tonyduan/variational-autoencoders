### Variational Autoencoder

---

Variational autoencoders model a latent variable <img alt="$z$" src="svgs/f93ce33e511096ed626b4719d50f17d2.svg" align="middle" width="8.367621899999993pt" height="14.15524440000002pt"/> and an observed variable <img alt="$x$" src="svgs/332cc365a4987aacce0ead01b8bdcc0b.svg" align="middle" width="9.39498779999999pt" height="14.15524440000002pt"/> via two networks [1].

1. Encoder network; <img alt="$q_\phi(z|x)​$" src="svgs/8c4290cd764b7be62885f1f2fa0f1ace.svg" align="middle" width="51.17860604999999pt" height="24.65753399999998pt"/> 
2. Decoder network; <img alt="$p_\theta(x|z)​$" src="svgs/7b4b76719fd0c5230e3c2d4849ba0924.svg" align="middle" width="50.82199814999999pt" height="24.65753399999998pt"/>

The encoder network tries to approximate the intractable posterior <img alt="$p_\theta(z|x)$" src="svgs/8d064232b1495aa703d8d2bb1a19d3aa.svg" align="middle" width="50.82199814999999pt" height="24.65753399999998pt"/>.

The model is trained by maximizing the evidence lower bound via gradient descent.
<p align="center"><img alt="$$&#10;\mathcal{L}(\theta,\phi) = \mathbb{E}_{z\sim q_\phi(z|x)}[p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p_\theta(z)) \leq p_\theta(x)&#10;$$" src="svgs/2e6a68c5089ed72134a0cf0384848df5.svg" align="middle" width="427.3819407pt" height="18.639307499999997pt"/></p>
We use low-variance Monte Carlo gradients computed via the reparameterization trick.

#### Examples

See the `examples/` folder for examples. Below we show the learned latent representation and forward samples for a mixture of Gaussians, with a standard normal prior.

![ex_model](examples/ex_2d.png "Example model output")

#### References

[1] D. P. Kingma & M. Welling, Auto-Encoding Variational Bayes. *Proceedings of the 2nd International Conference on Learning Representations* (2014).
