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
p_\theta(x_0)=\int dx^{(1...T)} p_\theta(x^{(0...T)})
$$

In the original original paper, the integral is intractable, so the fomula transformation is show below.
Although not described in detail in the paper, I personally think that the formula transformation was performed using the forward process,
which has a known probability distribution, probably because the probability distribution of the reverse process can be complicated
and it is difficult to calculate the integral.

$$
\begin{align*}
p_\theta(x_0)&=\int dx^{(1...T)} p_\theta(x^{(0...T)}) \frac{q(x^{(1...T)}|x_0)}{q(x^{(1...T)}|x_0)} \\
&=\int dx^{(1...T)} q(x^{(1...T)}|x_0) \frac{p_\theta(x^{(0...T)})}{q(x^{(1...T)}|x_0)} \\
&=\int dx^{(1...T)} q(x^{(1...T)}|x_0) p_\theta(x^{(T)})\prod_{t=1}^T \frac{p_\theta(x^{(t-1)}|x^{(t)})}{q(x^{(t)}|x^{(t-1)})}
\end{align*}
$$

Training is performed by optimizing the usual variational bound on negative log likelihood.
Below equation has a upper bound provided by Jensen’s inequality,

$$
\begin{align*}
E[-\log p_\theta(x_0)]&=
E\Big[-\log \Big[ \int dx^{(1...T)} q(x^{(1...T)}|x_0) p(x^{(T)})\prod_{t=1}^T \frac{p_\theta(x^{(t-1)}|x^{(t)})}{q(x^{(t)}|x^{(t-1)})}\Big] \Big] \\
&\leq E_q\Big[-\log \Big[p(x^{(T)})\prod_{t=1}^T \frac{p_\theta(x^{(t-1)}|x^{(t)})}{q(x^{(t)}|x^{(t-1)})}\Big] \Big] \\
&\leq E_q\Big[-\log p(x^{(T)}) - \sum_{t\geq1} \log \frac{p_\theta(x^{(t-1)}|x^{(t)})}{q(x^{(t)}|x^{(t-1)})} \Big] =:L
\end{align*}
$$



<br></br>

## How to create an environment

```bash
conda create -n diff python=3.11
conda activate diff
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
