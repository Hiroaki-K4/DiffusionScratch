# simple_diffusion

<br></br>

## Derivation of loss function
I show the derivatino of the loss function of the diffusion model.

First, difussion model has forward process and the reverse process. In the forward process, noise is added to the input data step by step. In the reverse process, the reverse of the forward process is performed to recover the original image from the noisy data. The graph below is easy to understand these processes.

<img src="images/forward_reverse.png" width='600'>



<br></br>

## How to create an environment

```bash
conda create -n diff python=3.11
conda activate diff
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
