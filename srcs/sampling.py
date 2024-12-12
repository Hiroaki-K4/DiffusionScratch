import matplotlib.pyplot as plt
import torch

from forward_process import calculate_parameters
from simple_nn import SimpleNN


def sampling(model_path, sample_num, diffusion_steps, min_beta, max_beta):
    model = SimpleNN()  # Use the same architecture as before
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    x_init = torch.randn(size=(sample_num, 2))
    print(x_init)
    print(x_init.shape)
    # plt.scatter(x_init[:,0], x_init[:,1], s=5)
    # plt.show()

    # print(model)
    beta_ts, alpha_ts, bar_alpha_ts = calculate_parameters(
        diffusion_steps, min_beta, max_beta
    )
    print(bar_alpha_ts)
    denoised_x = torch.zeros((diffusion_steps, x_init.shape[0], x_init.shape[1]))
    denoised_x[-1] = x_init
    # TODO: Fix result
    for t in range(diffusion_steps - 1, 0, -1):
        if t > 1:
            z = torch.randn(x_init.shape)
        else:
            z = 0
        ts = torch.full((x_init.shape[0], 1), t)
        mu = (
            1
            / torch.sqrt(alpha_ts[t])
            * (
                (denoised_x[t] - (1 - alpha_ts[t]) / torch.sqrt(1 - bar_alpha_ts[t]))
                * model.forward(denoised_x[t], ts)
            )
        )
        denoised_x[t - 1] = mu + torch.sqrt(beta_ts[t]) * z

    print(denoised_x)
    plt.scatter(
        denoised_x[0].detach().numpy()[:, 0], denoised_x[0].detach().numpy()[:, 1], s=5
    )
    plt.show()


if __name__ == "__main__":
    model_path = "diffusion_model.pth"
    sample_num = 100000
    diffusion_steps = 50
    min_beta = 1e-4
    max_beta = 0.02
    sampling(model_path, sample_num, diffusion_steps, min_beta, max_beta)
