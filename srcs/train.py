import torch
import torch.nn as nn

from forward_process import calculate_parameters
from prepare_dataset import create_original_data
from simple_nn import SimpleNN


def train(data, batch_size, device, epochs, diffusion_steps, min_beta, max_beta):
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    model = SimpleNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    bar_alpha_ts = calculate_parameters(diffusion_steps, min_beta, max_beta)
    print(bar_alpha_ts)
    for epoch in range(epochs):
        for x in data_loader:
            print(x.shape)
            input()


if __name__ == "__main__":
    batch_size = 128
    epochs = 100
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using {0} device".format(device))

    sample_num = 100000
    noise_std = 0.5
    x = create_original_data(sample_num, noise_std)
    data = torch.tensor(x, dtype=torch.float32)
    # Normalizatin -> [-1, 1]
    data = (data - data.min(0).values) / (data.max(0).values - data.min(0).values)
    data = 2 * data - 1
    batch_size = 128
    epochs = 100
    diffusion_steps = 50
    min_beta = 1e-4
    max_beta = 0.02
    train(data, batch_size, device, epochs, diffusion_steps, min_beta, max_beta)
