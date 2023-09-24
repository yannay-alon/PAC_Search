import torch
from torch import nn
from torch import distributions

from matplotlib import pyplot as plt


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


# </editor-fold>


class NeuralDistribution(nn.Module):
    def __init__(self, kernel: Kernel, distribution: type[distributions.Distribution]):
        super().__init__()

        self.kernel = kernel

        assert distribution.has_rsample
        self.distribution = distribution

        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

    def forward(self, heuristic_value: torch.Tensor, sample: bool = False):
        mean, std = torch.chunk(self.network(heuristic_value), 2, dim=-1)

        # TODO: Compare different standard deviation transformations
        # std = torch.exp(std)
        # std = torch.relu(std)
        # std = torch.square(std)
        std = torch.abs(std)

        if sample:
            distribution = self.distribution(mean, std)
            target_value = distribution.rsample()
            return self.kernel(target_value) * heuristic_value
        else:
            return mean, std

    def prob(self, heuristic_value: torch.Tensor, target_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the CDF and log_prob of the target value given the heuristic value

        :param heuristic_value: The input for the network
        :param target_value: The target value
        :return:
        """
        mean, std = self(heuristic_value)

        distribution = self.distribution(mean, std)
        value = self.kernel.inverse(target_value / heuristic_value)
        return distribution.cdf(value), distribution.log_prob(value)

    def plot(self, heuristic_values: torch.Tensor,
             n_samples: int = 1000, quantiles: tuple[float, float] = (0.25, 0.75)):
        with torch.no_grad():
            mean, std = self(heuristic_values.unsqueeze(dim=-1))
            heuristic_values = heuristic_values.cpu()
            mean = mean.cpu().squeeze(dim=-1)
            std = std.cpu().squeeze(dim=-1)

        distribution = self.distribution(mean, std)

        values = distribution.sample(torch.Size([n_samples]))
        target_values = self.kernel(values) * heuristic_values
        log_prob = distribution.log_prob(values)

        plt.scatter(heuristic_values.expand(n_samples, -1), target_values, s=0.1, c=log_prob, cmap="viridis")

        std, mean = torch.std_mean(target_values, dim=0)
        lower, upper = torch.quantile(target_values, torch.tensor(quantiles), dim=0)

        plt.plot(heuristic_values, mean, color="red")
        plt.fill_between(heuristic_values, lower, upper, alpha=0.5, color="red")

        plt.title(f"Neural Distribution: {self.distribution.__name__}, {self.kernel.__class__.__name__}")
        plt.xlabel("Heuristic Value")
        plt.ylabel("Estimated Optimal Value")
        plt.show()


def main():
    neural_distribution = NeuralDistribution(SoftPlus(), distributions.Normal)

    neural_distribution.plot(torch.linspace(1e-6, 10, 1000))


if __name__ == '__main__':
    main()
