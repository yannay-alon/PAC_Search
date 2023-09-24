import os
import time
import itertools
import numpy as np
import pandas as pd
from print_color import print

from puzzle import State, PuzzleProblem
from algorithms import WeightedAStar, AnytimeWeightedAStar, ParametricAnytimeWeightedAStar


def performances():
    s = 3
    repetitions = 100
    problem = PuzzleProblem(size=s)

    approximation_errors = [0.1, 0.5]
    required_confidences = [0.2, 0.8, 0.9]
    weights = [1.5, 2, 4]
    distribution_types = ["beta", "norm", None]

    ratio_column = "Anytime A*"
    df = pd.read_csv(f"data/puzzle_{s}/solution_lengths.csv")
    df = df[["initial_heuristic", ratio_column]].dropna()
    ratios = df["initial_heuristic"] / df[ratio_column]

    candidate_algorithms = []
    for approximation_error, required_confidence, weight in itertools.product(approximation_errors,
                                                                              required_confidences, weights):
        try:
            candidate_algorithms.append(
                AnytimeWeightedAStar(problem, weight, approximation_error, required_confidence, ratios))
        except AssertionError as exception:
            print(exception, tag="WARNING", tag_color="yellow")

    # for approximation_error, required_confidence, weight, distribution_type in itertools.product(
    #         approximation_errors, required_confidences, weights, distribution_types):
    #     try:
    #         candidate_algorithms.append(
    #             ParametricAnytimeWeightedAStar(problem, weight, approximation_error, required_confidence, ratios,
    #                                            distribution_type=distribution_type))
    #     except AssertionError as exception:
    #         print(exception, tag="WARNING", tag_color="yellow")
    #
    # for algorithm in candidate_algorithms:
    #     algorithm.dist.check_fit()

    results = {
        algorithm.parameters: {
            "success": [],
            "expanded_nodes_counts": [],
        }
        for algorithm in candidate_algorithms
    }

    # A* algorithm (with open-set tracking)
    a_star_parameters = (0, 1, 1)
    results[a_star_parameters] = {"success": [], "expanded_nodes_counts": []}
    # optimal_solver = AnytimeWeightedAStar(problem)
    optimal_solver = WeightedAStar(problem)

    for i in range(repetitions):
        print(f"\nStart of problem {i + 1}/{repetitions}", format="bold")
        position = State.get_random_state(s).position
        position = np.array([
            [0, 1, 6],
            [7, 8, 2],
            [4, 5, 3]
        ])
        problem.reset(State(position))

        optimal_score = None
        for solution in optimal_solver.search():
            optimal_score = solution[-1].depth
        # print(solution)

        results[(0, 1, 1)]["expanded_nodes_counts"].append(problem.expanded_nodes)
        results[(0, 1, 1)]["success"].append(True)

        for algorithm in candidate_algorithms:
            print(f"\t{algorithm}:")
            solution_score = None
            for solution in algorithm.search():
                solution_score = solution[-1].depth

            success = solution_score <= (1 + algorithm.approximation_error) * optimal_score
            if success:
                print(f"Score: {solution_score}, Optimal: {optimal_score}", tag="SUCCESS", tag_color="green")
            else:
                print(f"Score: {solution_score}, Optimal: {optimal_score}", tag="FAIL", tag_color="red")
            results[algorithm.parameters]["success"].append(success)
            results[algorithm.parameters]["expanded_nodes_counts"].append(problem.expanded_nodes)

    df = pd.DataFrame.from_dict(
        {
            algorithm_name: (np.mean(values["success"]), np.mean(values["expanded_nodes_counts"]))
            for algorithm_name, values in results.items()
        }
    )

    df.to_csv(f"data/results_based_{ratio_column}_test.csv", index=False)


def estimate_heuristics():
    s = 4
    repetitions = 1

    problem = PuzzleProblem(size=s)

    candidate_algorithms = [
        # AnytimeWeightedAStar(problem, 1),

        AnytimeWeightedAStar(problem, 1.25),

        AnytimeWeightedAStar(problem, 1.5),

        AnytimeWeightedAStar(problem, 2),
    ]

    columns = ["run_times", "expanded_nodes_counts", "solution_lengths"]
    statistics = {
        str(algorithm): {
            column: [] for column in columns
        } for algorithm in candidate_algorithms
    }

    initial_heuristics = []

    for i in range(repetitions):
        print(f"\nStart of problem {i + 1}/{repetitions}", format="bold")
        position = State.get_random_state(s).position
        problem.reset(State(position))

        initial_heuristic = problem.heuristic(problem.initial_state)
        initial_heuristics.append(initial_heuristic)

        for algorithm in candidate_algorithms:
            print(f"\t{algorithm}:")

            start_time = time.perf_counter()
            solution = None
            for solution in algorithm.search():
                break
            end_time = time.perf_counter()

            statistics[str(algorithm)]["run_times"].append(end_time - start_time)
            statistics[str(algorithm)]["expanded_nodes_counts"].append(problem.expanded_nodes)
            statistics[str(algorithm)]["solution_lengths"].append(len(solution) - 1)

            print(f"\t\t{len(solution) - 1} steps")
            if solution is not None:
                print(f"\t\tFinished in {end_time - start_time:.3f} seconds", tag="Success", tag_color="green")
            else:
                print(f"\t\t", tag="Fail", tag_color="red")

    print("\nResults:", format="underline")
    for algorithm in candidate_algorithms:
        print(f"\t{algorithm}:")

        for measure_name, measure_value in statistics[str(algorithm)].items():
            print(f"\t\t{measure_name}: {np.mean(measure_value):,.3f} ± {np.std(measure_value):,.3f}")

    os.makedirs(f"data/puzzle_{s}", exist_ok=True)

    for column in columns:
        df = pd.DataFrame.from_dict({str(algorithm): statistics[str(algorithm)][column]
                                     for algorithm in candidate_algorithms})
        if column == "solution_lengths":
            df["initial_heuristic"] = initial_heuristics

        df.to_csv(f"data/puzzle_{s}/{column}.csv", index=False)


if __name__ == '__main__':
    program_start_time = time.perf_counter()
    # estimate_heuristics()
    performances()
    program_end_time = time.perf_counter()
    print(f"\tProgram finished in {program_end_time - program_start_time:.3f} seconds")
