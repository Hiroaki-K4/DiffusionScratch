from sklearn.datasets import make_swiss_roll


def create_original_data(sample_num, noise_std):
    x, _ = make_swiss_roll(n_samples=sample_num, noise=noise_std)
    # Extract x and z axis
    x = x[:, [0, 2]]
    # Normalize value
    x = (x - x.mean()) / x.std()

    return x
