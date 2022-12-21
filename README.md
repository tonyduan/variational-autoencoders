### Variational Autoencoders

Last update: December 2022.

---

The VAE models a latent variable $z$ and an observed variable $x$.

```mermaid
graph LR;
    Z((Z)) -->X((X));
```

We assume a forward model parameterized by $\theta$, and a backward "approximate posterior" model by $\phi$.
```math
\begin{align*}
p_\theta(x, z) & = p_\theta(z)p_\theta(x|z) & q_\phi(z|x) & \approx p_\theta(z|x)
\end{align*}
```
To explain this, it's easier to examine the general case of maximizing the marginal likelihood in a latent-variable model.

**The Evidence Lower Bound**

Consider maximizing the likelihood of observed variables $x$ with latent variables $z$, with a model $\theta$.

This comes up in many situations, not only in the context of VAEs. Examples:

- Clustering: $x$ is observations, $z$ are membership identities.
- Bayesian inference: $x$ is observations, $z$ are model parameters.

It's easy to hypothesize a model $p_\theta(x|z)$. But typically a closed-form model for the *posterior* of the latent variables is intractable due to the dimensionality of $z$. That is, it's impossible to compute the following.
```math
p_\theta(z|x)=\frac{p(x,z)}{\int_zp(x,z)dz}
```
What can we do?

**Property 1**. The Evidence Lower Bound (ELBO), soon to be defined, is a valid lower bound on the marginal likelihood.

Thanks to Jensen's inequality we have the following result on the marginal likelihood.
```math
\begin{align*}
\log p_\theta(x) & = \log \int_z p_\theta(x,z)dz\\
& = \log \int_zp_\theta(z)p_\theta(x|z)dz\\
& \geq \int_z p_\theta(z)\log p_\theta(x|z)dz\\
& =\mathbb{E}_{z\sim p_\theta(z)}[\log p_\theta(x|z)]\\
\end{align*}
```

Following similar logic, there is a bound that arises if we plug in *any* valid distribution $q_\phi(z|x)$ with parameters $\phi$. (Personally I find it easier to understand the equations below with $q_\phi(z|x)$ substituted with $q_\phi(z)$ i.e. an unconditional distribution. It doesn't matter whether it's conditioned on $x$ or not.)
```math
\begin{align*}
\log p_\theta(x) & = \log \int_z q_\phi(z|x)\frac{p_\theta(x,z)}{q_\phi(z|x)}dz\\
& \geq \int_z q_\phi(z|x) \log \frac{p_\theta(x,z)}{q_\phi(z|x)}dz\\
& \triangleq \mathcal{L}_{\theta,\phi}(x)\\
& = \mathbb{E}_{z\sim q_\phi(z|x)}[\log p_\theta(x,z)]+ H(q_\phi(z|x)) \\
& = \mathbb{E}_{z\sim q_\phi(z|x)}[\log p_\theta(x|z)]+\mathbb{E}_{z\sim q_\phi(z|x)}[\log p_\theta(z)]+H(q_\phi(z|x))\\
&= \mathbb{E}_{z\sim q_\phi(z|x)}[\log p_\theta(x|z)] -D_\mathrm{KL}(\ q_\phi(z|x)\ \|\ p_\theta(z)\ )
\end{align*}
```
The last three slines are all equivalent forms of the ELBO.

**Property 2**. The gap between the marginal likelihood and the ELBO is exactly the KL divergence between the true intractable posterior $p_\theta(z|x)$ and the approximated posterior $q_\phi(z|x)$.

Let's compute the gap between the marginal likelihood and the ELBO.
```math
\begin{align*}
\log p_\theta(x) - \mathcal{L}_{\theta,\phi}(x)
& = \log \int_z p_\theta(x,z) dz -  \int_z q_\phi(z|x) \log \frac{p_\theta(x)p_\theta(z|x)}{q_\phi(z|x)}dz\\
& = \log \int_z p_\theta(z|x) dz + \log p_\theta(x) -  \int_z q_\phi(z|x) \log \frac{p_\theta(z|x)}{q_\phi(z|x)}dz - \log p_\theta(x)\\
& = \log \int_z p_\theta(z|x) dz  -  \int_z q_\phi(z|x) \log \frac{p_\theta(z|x)}{q_\phi(z|x)}dz \\
& = D_{KL}(\ q_\phi(z|x)\ \|\ p_\theta(z|x)\ ) \geq 0
\end{align*}
```
So by maximizing the ELBO, we're actually optimizing how well we approximate the intractable posterior!

Note that by invoking the non-negativity of KL divergence, this derivation yields another proof of Property 1 (though a less intuitive way to achieve the same result).

**The Reparameterization Trick**

A traditional variational auto-encoder makes the following choices:

1. The distribution $p_\theta(x|z) \sim N(\mu,\Sigma)$ where $\mu,\Sigma$ are output by an "encoder" neural network dependent on $z$.
2. The distribution $q_\phi(z|x) \sim N(\mu,\Sigma)$ where $\mu,\Sigma$ are output by a "decoder" neural network dependent on $x$.
3. The distribution $p_\theta(z) \sim N(0,I)$  and is typically fixed and not learned.

There are a variety of noise assumptions on $\Sigma$ we can choose, in this codebase we fully factorized covariance matrices throughout i.e. every predicted $\Sigma  = \mathrm{diag}(\sigma^2_d)$. For details see [[this repository]](https://github.com/tonyduan/mdn#mixture-density-network).

To optimize we perform gradient descent on the last of the equivalent versions of the ELBO.
```math
\mathcal{L}_{\theta,\phi}(x) = \mathbb{E}_{z\sim q_\phi(z|x)}[\log p_\theta(x|z)] - D_\mathrm{KL}(q_\phi(z|x)\ \|\ p_\theta(z))
```
The KL divergence term has a closed form solution (between two Gaussians), and therefore a closed-form gradient. But it's not obvious how to best compute the reconstruction term for gradient descent.

There are two "tricks" to do so. Let's again look at the general case first: our goal is to compute
```math
\begin{align*}
\nabla_\theta \mathbb{E}_{x\sim p_\theta(x)}[f(x)]
\end{align*}
```
**Trick 1.** The log-derivative trick (aka "REINFORCE"). Recall that for any function $g$,
```math
\nabla_x \log g(x) = \frac{\nabla_x g(x)}{g(x)}
```
Then we can derive the following Monte Carlo gradient estimate.
```math
\begin{align*}
\nabla_\theta \mathbb{E}_{x \sim p_\theta(x)}[f(x)] & = \nabla_\theta \int_xp_\theta(x) f_(x)dx\\

& = \int_xp_\theta(x)\frac{\nabla_\theta p_\theta(x)}{p_\theta(x)}f(x)dx\\
& = \int_xp_\theta(x)\nabla_\theta \log p_\theta(x) f(x) dx\\
& = \mathbb{E}_{x\sim p_\theta(x)}[\nabla_\theta \log p_\theta(x) f(x)]\\
& \approx \frac{1}{L}\sum_{l=1}^L \nabla_\theta \log p_\theta(x^{(l)}) f(x^{(l)})
\end{align*}
```
**Trick 2**. The reparameterization trick. Suppose we can re-write the sampling process as
```math
\begin{align*}
x & \sim p_\theta(x) & & \iff&  x = g_\theta(\epsilon), \epsilon \sim p(\epsilon).
\end{align*}
```
For example, in the case of a Normal distribution it's well known that
```math
\begin{align*}
x & \sim N(\mu,\Sigma) & & \iff&  x = L\epsilon + \mu, \epsilon \sim N(0,I) \text{ where } LL^\top = \Sigma.
\end{align*}
```
Then we can derive the following Monte Carlo gradient estimate.
```math
\begin{align*}
\nabla_\theta \mathbb{E}_{x\sim p_\theta(x)}[f(x)] & = \nabla_\theta \mathbb{E}_{\epsilon \sim p(\epsilon)}[f(g_\theta(\epsilon))]\\
&= \mathbb{E}_{\epsilon \sim p(\epsilon)}[\nabla_\theta f(g_\theta(\epsilon))]\\
& \approx \frac{1}{L}\sum_{l=1}^L \nabla_\theta f(g_\theta(\epsilon^{(l)}))
\end{align*}
```
The use of this estimator was the primaryinnovation of [1].

Putting it together in the context of VAEs, we can write the gradient as the following.
```math
\begin{align*}
\nabla_{\theta,\phi}\mathcal{L}_{\theta,\phi}(x)& =\nabla_{\theta,\phi} \left(\mathbb{E}_{z\sim q_\phi(z|x)}[\log p_\theta(x|z)] - D_\mathrm{KL}(\ q_\phi(z|x)\ \|\ p_\theta(z)\ )\right)\\
& = \mathbb{E}_{\epsilon\sim p(\epsilon)}[\nabla_{\theta,\phi}\log p_\theta(x|g_\phi(\epsilon))]- \nabla_{\theta,\phi}D_{KL}(\ q_\phi(z|x)\ \|\ p_\theta(z) \ )
\end{align*}
```

#### Examples

See the `examples/` folder for examples. Below we show the learned latent representation and forward samples for a mixture of Gaussians, with a standard normal prior.

![ex_model](examples/ex_2d.png "Example model output")

Additionally we try to reconstruct digits.

![ex_model](examples/ex_digits.png "Example model output")

#### References

[1] D. P. Kingma & M. Welling, Auto-Encoding Variational Bayes. *International Conference on Learning Representations* (2014).
