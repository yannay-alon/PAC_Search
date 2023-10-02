import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from typing import Optional

supported_distribution_types = {
    "beta": stats.beta,
    "gamma": stats.gamma,
    "norm": stats.norm,
    "uniform": stats.uniform,
    "cauchy": stats.cauchy,
    "alpha": stats.alpha,
}


class Distribution:
    def __init__(self, data: np.ndarray, distribution_type: Optional[str] = None, bounds: tuple[tuple[int]] = None):
        self.data = data
        self.dist = None
        self.model = None
        self.params = None
        if distribution_type is not None:
            try:
                self.dist = supported_distribution_types[distribution_type]
                bounds = ((-1e2, 1e2), (-1e2, 1e2)) if bounds is None else bounds
                self.model = stats.fit(self.dist, self.data, bounds)
                self.params = self.model.params
            except KeyError:
                print("Distribution type not supported. Using nonparametric empirical cdf.")
        else:
            print("Distribution type not specified. Using nonparametric empirical cdf.")

    def cdf(self, x: float, xp: float = None, fp: float = None):
        if self.dist is None:
            return np.interp(x, xp, fp)
        return self.dist.cdf(x, *self.params)

    def check_fit(self, plot=True):
        if self.dist is None:
            return

        # res = stats.goodness_of_fit(self.dist, self.data)
        # print(f"ks_test {res.statistic} p_value {res.pvalue}")

        if plot:
            self.model.plot()
            plt.show()


def plot_distribution(ratios: np.ndarray, algorithm_name: str):
    density, bins = np.histogram(ratios, density=True)
    unity_density = density / density.sum()

    fig, ax_1 = plt.subplots()
    ax_2 = ax_1.twinx()

    widths = bins[:-1] - bins[1:]
    ax_1.bar(bins[1:], density, width=0.3 * widths, color="blue", label="PDF")
    ax_2.plot(bins[1:], unity_density.cumsum(), color="tab:red", marker="s", label="CDF")

    plt.grid(axis="y")

    plt.title(f"PDF and CDF of {algorithm_name}")
    ax_1.set_xlabel(r"$h(s) \, / \, h^\circ(s)$ Ratio")
    ax_1.set_ylabel("PDF")
    ax_2.set_ylabel("CDF")
    plt.show()

    print()


def compare_distributions(df: pd.DataFrame):
    for algorithm in df.columns[:1]:
        plt.hist(df["initial_heuristic"] / df[algorithm], label=algorithm, density=True, alpha=0.5,
                 bins=20)
    plt.title(f"PDF")
    plt.legend()
    plt.show()

    for algorithm in df.columns[:1]:
        plt.hist(df["initial_heuristic"] / df[algorithm], label=algorithm, density=True, cumulative=True,
                 alpha=0.5, bins=20)
    plt.title(f"CDF")
    plt.legend()
    plt.show()


def main():
    s = 3
    df = pd.read_csv(f"data/puzzle_{s}/solution_lengths.csv")

    for column in df.columns[:-1]:
        plot_distribution(df["initial_heuristic"] / df[column], column)


if __name__ == '__main__':
    main()
