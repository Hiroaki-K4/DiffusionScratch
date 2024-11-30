import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.datasets import make_swiss_roll


def create_original_data():
    x, _ = make_swiss_roll(n_samples=100000, noise=0.5)
    # Extract x and z axis
    x = x[:, [0, 2]]
    # Normalize value
    x = (x - x.mean()) / x.std()

    return x


def add_noise(x):
    X = torch.tensor(x, dtype=torch.float32)

    # Calculate beta
    diffusion_steps = 40  # Number of steps in the diffusion process
    min_beta = 10e-4
    max_beta = 0.02
    step = (max_beta - min_beta) / diffusion_steps
    beta_ts = torch.range(min_beta, max_beta, step)
    print(beta_ts)
    sns.lineplot(beta_ts)
    plt.xlabel("Diffusion step")
    plt.ylabel(r"$\bar{\alpha}$")

    # Add noise
    # TODO: Implement forward process


if __name__ == "__main__":
    x = create_original_data()
    # plt.scatter(x[:, 0], x[:, 1])
    # plt.show()
    add_noise(x)
    # plt.show()
