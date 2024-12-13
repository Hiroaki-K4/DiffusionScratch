import torch
import torch.nn as nn

from forward_process import calculate_data_at_certain_time, calculate_parameters
from prepare_dataset import create_original_data
from simple_nn import SimpleNN


def train(
    data,
    batch_size,
    device,
    epochs,
    diffusion_steps,
    min_beta,
    max_beta,
    learning_rate,
    output_model_path,
):
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    model = SimpleNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    beta_ts, alpha_ts, bar_alpha_ts = calculate_parameters(
        diffusion_steps, min_beta, max_beta
    )
    for epoch in range(epochs):
        count = 0
        epoch_loss = 0
        for x in data_loader:
            random_time_step = torch.randint(0, diffusion_steps, size=[len(x), 1])
            noised_x_t, eps = calculate_data_at_certain_time(
                x, bar_alpha_ts, random_time_step
            )
            predicted_eps = model.forward(
                noised_x_t.to(device), random_time_step.to(device)
            )
            loss = loss_fn(predicted_eps, eps.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            count += 1

        print("Epoch {0}, Loss={1}".format(epoch, round(epoch_loss / count, 5)))

    print("Finished training!!")
    torch.save(model.state_dict(), output_model_path)
    print("Saved model: ", output_model_path)


if __name__ == "__main__":
    batch_size = 128
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
    batch_size = 128
    epochs = 30
    diffusion_steps = 50
    min_beta = 1e-4
    max_beta = 0.02
    learning_rate = 1e-3
    output_model_path = "diffusion_model.pth"
    train(
        data,
        batch_size,
        device,
        epochs,
        diffusion_steps,
        min_beta,
        max_beta,
        learning_rate,
        output_model_path,
    )
