# simple_diffusion

<br></br>

## Derivation of loss function
I show the derivatino of the loss function of the diffusion model.

First, difussion model has forward process and the reverse process. In the forward process, noise is added to the input data step by step.
In the reverse process, the reverse of the forward process is performed to recover the original image from the noisy data.
The graph below is easy to understand these processes.

<img src="images/forward_reverse.png" width='600'>

Diffusion model are latent variable models the form $p_\theta(x_0):=\int p_\theta(x_{0:T})dx_{1:T}$.
The joint distribution $p_\theta(x_{0:T})$ is called the reverse process,
and it is defined as a Markov chain with learned Gaussian transitions starting at $p(x_T)=\mathcal{N}(x_T;0,I)$:

$$
p_\theta(x_{0:T}):=p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}|x_t), \quad p_\theta(x_{t-1}|x_t):=\mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))
$$

Forward process is fixed to a Markov chain that gradually adds Gaussian noise to the data according to a variance schedule $\beta_1,...,\beta_T$:

$$
q(x_{1:T}|x_0):=\prod_{t=1}^T q(x_t|x_{t-1}), \quad q(x_t|x_{t-1}):=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t I)
$$

The probability the generative model assigns to the data is as follows.

$$
p_\theta(x_0)=\int dx_{1:T} p_\theta(x_{0:T})
$$

In the original original paper, the integral is intractable, so the fomula transformation is show below.
Although not described in detail in the paper, I personally think that the formula transformation was performed using the forward process,
which has a known probability distribution, probably because the probability distribution of the reverse process can be complicated
and it is difficult to calculate the integral.

$$
\begin{align*}
p_\theta(x_0)&=\int dx_{1:T} p_\theta(x_{0:T}) \frac{q(x_{1:T}|x_0)}{q(x_{1:T}|x_0)} \\
&=\int dx_{1:T} q(x_{1:T}|x_0) \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \\
&=\int dx_{1:T} q(x_{1:T}|x_0) p_\theta(x_T)\prod_{t=1}^T \frac{p_\theta(x_{t-1}|x_{t})}{q(x_{t}|x_{t-1})}
\end{align*}
$$

Training is performed by optimizing the usual variational bound on negative log likelihood.
Below equation has a upper bound provided by Jensen’s inequality,

$$
\begin{align*}
E[-\log p_\theta(x_0)]&=
E\Big[-\log \Big[ \int dx_{1:T} q(x_{1:T}|x_0) p(x_{T})\prod_{t=1}^T \frac{p_\theta(x_{t-1}|x_{t})}{q(x_{t}|x_{t-1})} \Big] \Big] \\
&\leq E_q\Big[-\log \Big[p(x_{T})\prod_{t=1}^T \frac{p_\theta(x_{t-1}|x_{t})}{q(x_{t}|x_{t-1})}\Big] \Big] \\
&\leq E_q\Big[-\log p(x_{T}) - \sum_{t\geq1} \log \frac{p_\theta(x_{t-1}|x_{t})}{q(x_{t}|x_{t-1})} \Big] =:L
\end{align*}
$$

The equation can be further transformed as follows.

$$
\begin{align*}
L &= E_q\Big[ -\log p(x_{T}) - \sum_{t\geq1} \log \frac{p_\theta(x_{t-1}|x_{t})}{q(x_{t}|x_{t-1})} \Big] \\
&= E_q \Big[ -\log p(x_{T}) - \sum_{t>1} \log \frac{p_\theta(x_{t-1}|x_{t})}{q(x_{t}|x_{t-1})} - \log \frac{p_\theta(x_0|x_1)}{q(x_1|x_0)} \Big] \\
&= E_q \Big[ -\log p(x_{T}) - \sum_{t>1} \log \frac{p_\theta(x_{t-1}|x_{t})}{q(x_{t-1}|x_{t},x_0)} \frac{q(x_{t-1}|x_0)}{q(x_t|x_0)} - \log \frac{p_\theta(x_0|x_1)}{q(x_1|x_0)} \Big] \\
&= E_q \Big[ -\log \frac{p(x_{T})}{q(x_T|x_0)} - \sum_{t>1} \log \frac{p_\theta(x_{t-1}|x_{t})}{q(x_{t-1}|x_{t},x_0)} - \log p_\theta(x_0|x_1) \Big] \\
&= E_q \Big[ D_{KL}(q(x_T|x_0) \parallel p(x_T)) + \sum_{t>1}D_{KL}(q(x_{t-1}|x_t,x_0)\parallel p_\theta (x_{t-1}| x_t)) - \log p_\theta(x_0|x_1) \Big] \\
&= E_q \Big[ L_T + \sum_{t>1}L_{t-1} + L_0 \Big]
\end{align*}
$$

In the above equation deformation, $q(x_t|x_{t-1})$ was transformed as follows.

$$
\begin{align*}
q(x_t|x_{t-1}) &= q(x_t|x_{t-1},x_0) \\
&= \frac{q(x_t,x_{t-1}|x_0)}{q(x_{t-1}|x_0)} \\
&= q(x_{t-1}|x_t,x_0) \frac{q(x_t|x_0)}{q(x_{t-1}|x_0)}
\end{align*}
$$

We also used the following equation transformation.

$$
\begin{align*}
\sum_{t>1} \frac{q(x_{t-1}|x_0)}{q(x_t|x_0)} \frac{1}{q(x_1|x_0)}
&= \frac{\cancel{q(x_{T-1}|x_0)}...\cancel{q(x_{1}|x_0)}}{q(x_{T}|x_0)...\cancel{q(x_{2}|x_0)}} \frac{1}{\cancel{q(x_1|x_0)}} \\
&= \frac{1}{q(x_{T}|x_0)}
\end{align*}
$$

$D_{KL}$ is called **KL Divergence** and is a type of statistical distance:
a measure of how one reference probability distribution $P$ is different from a second probability distribution $Q$.

$$
\begin{align*}
KL(p \parallel q) &= \int p(x) \log(q(x))dx - \Big(-\int p(x)\log p(x)dx \Big) \\
&= - \int p(x) \log \frac{q(x)}{p(x)} dx
\end{align*}
$$

### Forward process and $L_T$
We ignore the fact that the forward process variances $β_t$ are learnable by reparameterization and
instead ﬁx them to constants. Thus, in our implementation, the approximate posterior $q$ has no learnable parameters,
**so $L_T$ is a constant during training and can be ignored.**

### Reverse process and $L_{1:T-1}$

$$
q(x_{t-1}|x_t, x_0) = \mathcal{N} (x_{t-1};\tilde{\mu_t}(x_t,x_0), \tilde{\beta_t}I), \\
where \quad \tilde{\mu_t}(x_t,x_0) := \frac{\sqrt{\bar{\alpha_{t-1} \beta_t}}}{1-\bar{\alpha_{t}}} x_0 +
\frac{\sqrt{\alpha_t}(1-\bar{\alpha_{t-1}})}{1-\bar{\alpha_t}} x_t \quad and \quad \tilde{\beta_t}:=\frac{1-\bar{\alpha_{t-1}}}{1-\bar{\alpha_t}} \beta_t
$$

<br></br>

## Appendix
### A. Derivation of mean and variance of $q(x_{t-1}|x_t,x_0)$
The conditional distribution $q(x_{t-1}|x_t,x_0)$ is proportional to the product of the following two distributions.

$$
q(x_{t-1}|x_t,x_0) \propto q(x_t|x_{t-1}) q(x_{t-1}|x_0)
$$

Those two distributions are defined as follows.

$$
\begin{align*}
q(x_t|x_{t-1}) &= \mathcal{N} (x_t;\sqrt{\alpha_t} x_{t-1}, \beta_t I) \\
q(x_{t-1}|x_0) &= \mathcal{N} (x_{t-1};\sqrt{\bar{\alpha_{t-1}}} x_0, (1-\bar{\alpha_{t-1}}) I)
\end{align*}
$$

When calculating variance, we use the product property of the Gaussian distribution.
The inverse of the variance in a product of Gaussian distribution is expressed as
the sum of the inverse variances of the individual distributions.

$$
\begin{align*}
\frac{1}{\tilde{\beta_t}} &= \frac{1}{\beta_t} + \frac{1}{1-\bar{\alpha_{t-1}}} \\
&= \frac{1-\bar{\alpha_{t-1}}+\beta_t}{\beta_t (1-\bar{\alpha_{t-1}})}
\end{align*}
$$

If we inverse both sides to obtain the variance $\tilde{\beta_t}$

$$
\tilde{\beta_t} = \frac{\beta_t (1-\bar{\alpha_{t-1}})}{1 - \bar{\alpha_{t-1}} + \beta_t}
$$

Here, we use the following property.

$$
1 - \bar{\alpha_{t}} = (1 - \bar{\alpha_{t-1}}) + \beta_t
$$

Substitute this into the variance fomula to transforme it.

$$
\tilde{\beta_t} = \frac{\beta_t (1-\bar{\alpha_{t-1}})}{1-\bar{\alpha_t}}
$$

TODO: Add derivation of mean

<br></br>

## How to create an environment

```bash
conda create -n diff python=3.11
conda activate diff
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
