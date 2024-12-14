import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation

from prepare_dataset import create_original_data


def calculate_parameters(diffusion_steps, min_beta, max_beta):
    step = (max_beta - min_beta) / diffusion_steps
    beta_ts = torch.arange(min_beta, max_beta + step, step)

    alpha_ts = 1 - beta_ts
    bar_alpha_ts = torch.cumprod(alpha_ts, dim=0)

    return beta_ts, alpha_ts, bar_alpha_ts


def calculate_data_at_certain_time(x_0, bar_alpha_ts, t):
    eps = torch.randn(size=x_0.shape)
    noised_x_t = (
        torch.sqrt(bar_alpha_ts[t]) * x_0 + torch.sqrt(1 - bar_alpha_ts[t]) * eps
    )

    return noised_x_t, eps


def create_forward_process_animation(x, diffusion_steps, min_beta, max_beta, save_path):
    X = torch.tensor(x, dtype=torch.float32)
    beta_ts, alpha_ts, bar_alpha_ts = calculate_parameters(
        diffusion_steps, min_beta, max_beta
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter([], [], alpha=0.1, s=1)

    def init():
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Forward Process")
        return (scatter,)

    def update(t):
        noised_x_t, eps = calculate_data_at_certain_time(X, bar_alpha_ts, t)
        # Update scatter plot
        scatter.set_offsets(noised_x_t)
        ax.set_title(f"Forward Process - Step {t}/{diffusion_steps}")
        return (scatter,)

    # Create animation
    anim = FuncAnimation(fig, update, frames=diffusion_steps, init_func=init, blit=True)

    # Save animation as video
    anim.save(save_path, writer="pillow", fps=10)
    plt.close(fig)
    print("Finish saving gif file: ", save_path)


if __name__ == "__main__":
    sample_num = 100000
    noise_std = 0.5
    x = create_original_data(sample_num, noise_std)
    diffusion_steps = 50
    min_beta = 1e-4
    max_beta = 0.02
    save_path = "../resources/forward_process.gif"
    create_forward_process_animation(x, diffusion_steps, min_beta, max_beta, save_path)
