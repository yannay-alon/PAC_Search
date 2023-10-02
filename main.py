import os
import time
import itertools
import numpy as np
import pandas as pd
from print_color import print

from neural_distribution import *
from puzzle import State, PuzzleProblem
from algorithms import AnytimeWeightedAStar, ParametricAnytimeWeightedAStar, NeuralAnytimeWeightedAStar


def performances():
    s = 3
    repetitions = 300
    problem = PuzzleProblem(size=s)

    approximation_errors = [0.2, 0.4, 0.8]
    required_confidences = [0.5, 0.75, 0.8]
    weights = [2, 5, 10]
    distribution_types = ["beta", "norm", "gamma", "uniform", "cauchy", "alpha", None]
    distribution_types = ["beta", "norm", None]

    ratio_column = "Anytime 2-A*"
    num_samples = 10
    df = pd.read_csv(f"data/puzzle_{s}/solution_lengths.csv")
    df = df[["initial_heuristic", ratio_column]].dropna()
    ratios = df["initial_heuristic"] / df[ratio_column]
    ratios = ratios.sample(num_samples, replace=True).values

    candidate_algorithms = dict()

    path = sorted(os.listdir("checkpoints"), key=lambda x: os.path.getmtime(os.path.join("checkpoints", x)))[-1]
    path = os.path.join("checkpoints", path)
    neural_distribution_model = NeuralDistribution.load(path)

    for approximation_error, required_confidence, weight in itertools.product(
            approximation_errors, required_confidences, weights):
        try:
            algorithm = NeuralAnytimeWeightedAStar(problem, neural_distribution_model, weight,
                                                   approximation_error, required_confidence)
            candidate_algorithms[algorithm.parameters] = algorithm
        except AssertionError as exception:
            print(exception, tag="WARNING", tag_color="yellow")

        for distribution_type in distribution_types:
            try:
                algorithm = ParametricAnytimeWeightedAStar(problem, weight, approximation_error, required_confidence,
                                                           ratios, distribution_type)
                candidate_algorithms[algorithm.parameters] = algorithm
                algorithm.dist.check_fit()
            except AssertionError as exception:
                print(exception, tag="WARNING", tag_color="yellow")

    results = {
        name: {
            "success": [],
            "success_confidence": [],
            "expanded_nodes_counts": [],
            "expanded_nodes_counts_confidence": [],
        }
        for name, algorithm in candidate_algorithms.items()
    }

    # A* algorithm (with open-set tracking i.e. max-f-min)
    optimal_solver = ParametricAnytimeWeightedAStar(problem)

    results[optimal_solver.parameters] = {
        "success": [],
        "success_confidence": [],
        "expanded_nodes_counts": [],
        "expanded_nodes_counts_confidence": [],
    }

    for i in range(repetitions):
        print(f"\nStart of problem {i + 1}/{repetitions}", format="bold")
        position = State.get_random_state(s).position
        problem.reset(State(position))

        optimal_score = None
        for solution, stop_reason, max_f_min_history in optimal_solver.search():
            optimal_score = solution[-1].depth

        results[optimal_solver.parameters]["expanded_nodes_counts"].append(problem.expanded_nodes)
        results[optimal_solver.parameters]["success"].append(True)

        # for algorithm in candidate_algorithms:
        for name, algorithm in candidate_algorithms.items():
            print(f"\t{algorithm}:")
            solution_score = None
            stop_reason = None
            for solution, stop_reason, max_f_min_history in algorithm.search():
                solution_score = solution[-1].depth

            success = solution_score <= (1 + algorithm.approximation_error) * optimal_score
            if success:
                print(f"Score: {solution_score}, Optimal: {optimal_score}", tag="SUCCESS", tag_color="green")
            else:
                print(f"Score: {solution_score}, Optimal: {optimal_score}", tag="FAIL", tag_color="red")

            results[algorithm.parameters]["success"].append(success)
            results[algorithm.parameters]["expanded_nodes_counts"].append(problem.expanded_nodes)

            if stop_reason == "Achieved Confidence":
                results[algorithm.parameters]["success_confidence"].append(success)
                results[algorithm.parameters]["expanded_nodes_counts_confidence"].append(problem.expanded_nodes)

    df = pd.DataFrame.from_dict(
        {
            algorithm_name: tuple(np.mean(column_values) for column_values in values.values())
            for algorithm_name, values in results.items()
        }, orient="index", columns=results[optimal_solver.parameters].keys()
    )
    df[optimal_solver.parameters_names] = list(df.index.values)

    df.T.to_csv(f"data/puzzle_{s}/results_based_{ratio_column}_{num_samples}_neural.csv")


def collect_data():
    problem_size = 3
    repetitions = 10

    problem = PuzzleProblem(size=problem_size)

    candidate_algorithms = [
        AnytimeWeightedAStar(problem, 1),
        AnytimeWeightedAStar(problem, 1.25),
        AnytimeWeightedAStar(problem, 1.5),
        AnytimeWeightedAStar(problem, 2),
    ]

    file_path = f"data/puzzle_{problem_size}"
    os.makedirs(file_path, exist_ok=True)

    existing_indices = []
    folder = f"data/puzzle_{problem_size}"
    path = f"{folder}/solution_lengths.csv"
    if os.path.exists(path):
        df = pd.read_csv(path).dropna(axis="rows")
        existing_indices = df["problem_index"].values

    num_problems = repetitions
    task_status = pd.DataFrame(columns=[i + 1 for i in range(num_problems)],
                               index=[str(algorithm) for algorithm in candidate_algorithms], dtype=str)
    # Background color: red
    task_status[:] = "\033[41m\033[30m Pending \033[39m\033[49m"

    results = {}
    for column in ["run_times", "expanded_nodes_counts", "solution_lengths", "max_f_min_history"]:
        try:
            data_frame = pd.read_csv(f"{file_path}/{column}.csv", header=0, index_col="problem_index")
        except FileNotFoundError:
            data_frame = pd.DataFrame(columns=["problem_index"]).set_index("problem_index")

        for algorithm in candidate_algorithms:
            if str(algorithm) not in data_frame.columns:
                data_frame[str(algorithm)] = None

        results[column] = data_frame

    for i in range(repetitions):
        print(f"\nStart of problem {i + 1}/{repetitions}")
        initial_state = State.get_random_state(problem_size, exclude_indices=existing_indices)
        problem.reset(initial_state)

        initial_heuristic = problem.heuristic(initial_state)
        results["solution_lengths"].at[problem.index, "initial_heuristic"] = initial_heuristic

        for algorithm in candidate_algorithms:
            problem.reset(initial_state)

            # Background color: cyan
            task_status[i + 1][str(algorithm)] = "\033[46m\033[30m Working \033[39m\033[49m"
            print("\n\n")
            print(task_status.to_markdown(tablefmt="grid"))

            start_time = time.perf_counter()
            solution = None
            max_f_min_history = []
            for solution, stop_reason, max_f_min_history in algorithm.search():
                break  # Stop after the first solution
            end_time = time.perf_counter()

            run_time = end_time - start_time
            expanded_nodes_count = problem.expanded_nodes
            solution_length = len(solution) - 1

            results["run_times"].at[problem.index, str(algorithm)] = run_time
            results["expanded_nodes_counts"].at[problem.index, str(algorithm)] = expanded_nodes_count
            results["solution_lengths"].at[problem.index, str(algorithm)] = solution_length
            results["max_f_min_history"].at[problem.index, str(algorithm)] = str(max_f_min_history)

            # Background color: green
            task_status[i + 1][str(algorithm)] = "\033[42m\033[30m Finished \033[39m\033[49m"
            print("\n\n")
            print(task_status.to_markdown(tablefmt="grid"))

    for key, data_frame in results.items():
        data_frame.to_csv(f"{file_path}/{key}.csv", index=True)


if __name__ == '__main__':
    program_start_time = time.perf_counter()
    performances()
    # collect_data()
    program_end_time = time.perf_counter()
    print(f"\tProgram finished in {program_end_time - program_start_time:.3f} seconds")
