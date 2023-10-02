import os
import torch
import lightning
from torch import nn
from torch import distributions
from lightning.pytorch import loggers
from torch.utils.data import Dataset, DataLoader

import wandb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Optional, Callable

__all__ = [
    "NeuralDistribution",

    "get_distribution",
    "get_activation",
    "get_kernel",

    "Kernel",
    "Exponential",
    "SoftPlus",
    "ReLU",
]


# <editor-fold desc="KERNEL MODULES">
class Kernel(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, target_value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Exponential(Kernel):
    def forward(self, value: torch.Tensor):
        return 1 + torch.exp(value)

    def inverse(self, target_value: torch.Tensor):
        return torch.log(target_value - 1)


class SoftPlus(Kernel):
    def __init__(self):
        super().__init__()
        self.soft_plus = nn.Softplus()

    def forward(self, value: torch.Tensor):
        return 1 + self.soft_plus(value)

    def inverse(self, target_value: torch.Tensor):
        return torch.log(torch.exp(self.soft_plus.beta * (target_value - 1)) - 1) / self.soft_plus.beta


class ReLU(Kernel):
    def forward(self, value: torch.Tensor):
        return 1 + torch.relu(value)

    def inverse(self, target_value: torch.Tensor):
        return torch.relu(target_value - 1)


def get_kernel(kernel_name: str) -> Kernel:
    kernel_options = {
        "exponential": Exponential,
        "softplus": SoftPlus,
        "relu": ReLU,
    }
    kernel_name = kernel_name.lower()
    assert kernel_name in kernel_options, f"Unknown kernel: {kernel_name}"

    return kernel_options[kernel_name]()


# </editor-fold>


# <editor-fold desc="UTILITY FUNCTIONS">
def get_distribution(distribution_name: str) -> type[distributions.Distribution]:
    distribution_options = {
        "normal": distributions.Normal,
        "laplace": distributions.Laplace,
        "cauchy": distributions.Cauchy,
    }
    distribution_name = distribution_name.lower()
    assert distribution_name in distribution_options, f"Unknown distribution: {distribution_name}"

    return distribution_options[distribution_name]


def get_activation(activation_name: str) -> type[nn.Module]:
    activation_options = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "leakyrelu": nn.LeakyReLU,
    }
    activation_name = activation_name.lower()
    assert activation_name in activation_options, f"Unknown activation: {activation_name}"

    return activation_options[activation_name]


def get_std_kernel(std_kernel_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    std_kernel_options = {
        "exp": torch.exp,
        "relu": torch.relu,
        "square": torch.square,
        "abs": torch.abs,
    }
    std_kernel_name = std_kernel_name.lower()
    assert std_kernel_name in std_kernel_options, f"Unknown std kernel: {std_kernel_name}"

    return std_kernel_options[std_kernel_name]


# </editor-fold>


class SearchDataset(Dataset):
    def __init__(self, puzzle_size: int, ground_truth_column: str):
        data = pd.read_csv(f"data/puzzle_{puzzle_size}/solution_lengths.csv")
        self.data = data[["initial_heuristic", ground_truth_column]].dropna(axis="rows")
        lower_bounds = pd.read_csv(f"data/puzzle_{puzzle_size}/max_f_min_history.csv")
        self.lower_bounds = lower_bounds[ground_truth_column].dropna(axis="rows").apply(eval)

    def __getitem__(self, index: int):
        heuristic_value, ground_truth_value = self.data.iloc[index]
        lower_bounds = self.lower_bounds[index]
        lower_bound = np.random.choice(lower_bounds)
        # lower_bound = lower_bounds[-1]

        heuristic_value = torch.tensor(heuristic_value, dtype=torch.float)
        lower_bound = torch.tensor(lower_bound, dtype=torch.float)
        ground_truth_value = torch.tensor(ground_truth_value, dtype=torch.float)
        return heuristic_value, lower_bound, ground_truth_value

    def __len__(self):
        return len(self.data)


class NeuralDistribution(lightning.LightningModule):
    def __init__(self, kernel: Kernel, distribution: type[distributions.Distribution],
                 hidden_dimension: int = 100, num_hidden_layers: int = 4, activation: type[nn.Module] = nn.GELU,
                 std_kernel: Callable[[torch.Tensor], torch.Tensor] = torch.exp,
                 learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.kernel = kernel

        assert distribution.has_rsample, "The distribution must have a rsample method"
        self.distribution = distribution

        all_dims = [3] + [hidden_dimension] * num_hidden_layers + [2]
        layers = []
        for input_dim, output_dim in zip(all_dims[:-1], all_dims[1:]):
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(activation())
        layers.pop()

        # self.network = nn.Sequential(*layers)

        self.input_encoder, *layers, self.output_decoder = layers
        self.layers = nn.ModuleList(layers)

        self.residual_weight = nn.Parameter(torch.tensor(0.1))

        self.std_kernel = std_kernel

        self.learning_rate = learning_rate
        self._epsilon = 1e-6

        self.mse = nn.MSELoss()

    def forward(self, heuristic_value: torch.Tensor, lower_bound: torch.Tensor, sample: bool = False):
        input_tensor = torch.stack([heuristic_value, 1 / heuristic_value, lower_bound], dim=-1)
        # mean, std = torch.chunk(self.network(input_tensor), 2, dim=-1)

        input_tensor = self.input_encoder(input_tensor)
        for layer in self.layers:
            input_tensor = layer(input_tensor) + input_tensor * self.residual_weight
        input_tensor = self.output_decoder(input_tensor)
        mean, std = torch.chunk(input_tensor, 2, dim=-1)

        std = self.std_kernel(std)
        std = self._epsilon + std

        if sample:
            distribution = self.distribution(mean, std)
            target_value = distribution.rsample()
            return self.kernel(target_value) * heuristic_value
        else:
            return mean, std

    def prob(self, heuristic_value: torch.Tensor, lower_bound: torch.Tensor,
             target_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the CDF and log_prob of the target value given the heuristic value

        :param heuristic_value: The input for the network
        :param lower_bound: The current lower bound
        :param target_value: The target value
        :return:
        """
        mean, std = self(heuristic_value, lower_bound)

        distribution = self.distribution(mean, std)
        value = self.kernel.inverse(target_value / heuristic_value + self._epsilon)
        return distribution.cdf(value), distribution.log_prob(value)

    def plot(self, heuristic_values: torch.Tensor, lower_bounds: torch.Tensor,
             ground_truth_values: Optional[torch.Tensor] = None,
             n_samples: int = 1000,
             quantiles: tuple[float, float] = (0.25, 0.75)):
        order = torch.argsort(heuristic_values)
        lower_bounds = lower_bounds[order]
        heuristic_values = heuristic_values[order]
        if ground_truth_values is not None:
            ground_truth_values = ground_truth_values[order]

        with torch.no_grad():
            mean, std = self(heuristic_values, lower_bounds)
            heuristic_values = heuristic_values.cpu()
            lower_bounds = lower_bounds.cpu()
            mean = mean.cpu().squeeze(dim=-1)
            std = std.cpu().squeeze(dim=-1)

        distribution = self.distribution(mean, std)

        values = distribution.sample(torch.Size([n_samples]))
        target_values = self.kernel(values) * heuristic_values
        log_prob = distribution.log_prob(values)

        plt.scatter(heuristic_values.expand(n_samples, -1), target_values, s=0.1, c=log_prob, cmap="viridis")

        std, mean = torch.std_mean(target_values, dim=0)
        lower, upper = torch.quantile(target_values, torch.tensor(quantiles), dim=0)

        plt.plot(heuristic_values, mean, color="red", label="Prediction Mean")
        plt.fill_between(heuristic_values, lower, upper, alpha=0.5, color="red")

        if ground_truth_values is not None:
            plt.scatter(heuristic_values, ground_truth_values, s=0.5, marker="x", color="black", label="Ground Truth")

        plt.title(f"Neural Distribution: {self.distribution.__name__}, {self.kernel.__class__.__name__}")
        plt.xlabel("Heuristic Value")
        plt.ylabel("Estimated Optimal Value")
        plt.legend()
        plt.show()

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx):
        heuristic_value, lower_bound, target_value = batch
        cdf, log_prob = self.prob(heuristic_value, target_value, lower_bound)
        loss = -log_prob.mean()

        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx):
        heuristic_value, lower_bound, target_value = batch

        mean, std = self(heuristic_value, lower_bound)
        ratio_targets = target_value / heuristic_value

        ratio_mse = self.mse(mean, ratio_targets)

        self.log("Validation/ratio_MSE", ratio_mse)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def load(checkpoint: str):
        return NeuralDistribution.load_from_checkpoint(checkpoint)


def main():
    configuration = {
        "kernel": "ReLU",
        "distribution": "Normal",
        "std_kernel": "exp",

        "hidden_dim": 100,
        "num_hidden_layers": 10,
        "activation": "ReLU",

        "learning_rate": 0.001,
        "batch_size": 20,
    }

    wandb.init(project="neural-distribution", name="neural-distribution", config=configuration)
    config = wandb.config

    puzzle_size = 3

    dataset = SearchDataset(puzzle_size, "Anytime A*")
    train_data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    validation_data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    neural_distribution = NeuralDistribution(get_kernel(config.kernel),
                                             get_distribution(config.distribution),
                                             hidden_dimension=config.hidden_dim,
                                             num_hidden_layers=config.num_hidden_layers,
                                             activation=get_activation(config.activation),
                                             std_kernel=get_std_kernel(config.std_kernel),
                                             learning_rate=config.learning_rate)

    logger = loggers.WandbLogger(project="neural-distribution", name="neural-distribution")
    callback = lightning.pytorch.callbacks.ModelCheckpoint(
        monitor="Validation/ratio_MSE",
        dirpath="checkpoints",
    )

    trainer = lightning.Trainer(
        fast_dev_run=False,

        max_epochs=100,
        accelerator="cpu",

        logger=logger,
        callbacks=[callback],
        gradient_clip_val=1.0,
    )
    trainer.fit(neural_distribution, train_data_loader, validation_data_loader)

    print(neural_distribution.residual_weight)


def use_model():
    puzzle_size = 3
    path = sorted(os.listdir("checkpoints"), key=lambda x: os.path.getmtime(os.path.join("checkpoints", x)))[-1]
    path = os.path.join("checkpoints", path)
    print(path)

    model = NeuralDistribution.load_from_checkpoint(path)

    dataset = SearchDataset(puzzle_size, "Anytime A*")
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    heuristic_value, lower_bound, target_value = next(iter(data_loader))

    model.plot(heuristic_value, lower_bound, target_value)


if __name__ == '__main__':
    # main()
    use_model()
