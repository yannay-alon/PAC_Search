import os
import time
import numpy as np
import pandas as pd
from print_color import print
import multiprocessing as multiprocess

from algorithms import SearchAlgorithm, AnytimeWeightedAStar
from puzzle import PuzzleProblem, State

TASK_START = "START"
TASK_FINISH = "FINISH"


def results_handler(file_path: str, candidate_algorithms: list[SearchAlgorithm], num_problems: int, queue):
    os.makedirs(file_path, exist_ok=True)

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

    while True:
        status, (index, problem_index), algorithm_name, algorithm_result = queue.get()
        index = index + 1

        if status == TASK_START:
            # Background color: cyan
            task_status[index][algorithm_name] = "\033[46m\033[30m Working \033[39m\033[49m"
            print("\n\n")
            print(task_status.to_markdown(tablefmt="grid"))
            continue
        elif status == TASK_FINISH:
            # Background color: green
            task_status[index][algorithm_name] = "\033[42m\033[30m Finished \033[39m\033[49m"
            print("\n\n")
            print(task_status.to_markdown(tablefmt="grid"))

        if problem_index is None:
            break

        run_time, expanded_nodes_count, solution_length, max_f_min_history = algorithm_result

        if problem_index not in results["solution_lengths"].index:
            position = np.array(problem_index.split(",")).astype(int)
            size = int(np.sqrt(len(position)))
            state = State(position.reshape(size, size))
            problem = PuzzleProblem(initial_state=state)
            initial_heuristic = problem.heuristic(state)
            results["solution_lengths"].at[problem_index, "initial_heuristic"] = initial_heuristic

        results["run_times"].at[problem_index, algorithm_name] = run_time
        results["expanded_nodes_counts"].at[problem_index, algorithm_name] = expanded_nodes_count
        results["solution_lengths"].at[problem_index, algorithm_name] = solution_length
        results["max_f_min_history"].at[problem_index, algorithm_name] = str(max_f_min_history)

        for key, data_frame in results.items():
            data_frame.to_csv(f"{file_path}/{key}.csv", index=True)


def problem_solver(index: int, position: np.ndarray, algorithm: SearchAlgorithm, queue):
    algorithm.problem.reset(State(position))

    queue.put((TASK_START, (index, algorithm.problem.index), str(algorithm), None))

    start_time = time.perf_counter()
    solution = None
    max_f_min_history = []
    for solution, stop_reason, max_f_min_history in algorithm.search():
        break  # Stop after the first solution
    end_time = time.perf_counter()

    algorithm_result = (end_time - start_time, algorithm.problem.expanded_nodes, len(solution) - 1, max_f_min_history)

    queue.put((TASK_FINISH, (index, algorithm.problem.index), str(algorithm), algorithm_result))


def main():
    max_processes = 40

    problem_size = 3
    repetitions = 900

    problem = PuzzleProblem(size=problem_size)

    candidate_algorithms = [
        AnytimeWeightedAStar(problem, 1),
        AnytimeWeightedAStar(problem, 1.25),
        AnytimeWeightedAStar(problem, 1.5),
        AnytimeWeightedAStar(problem, 2),
    ]

    manager = multiprocess.Manager()
    queue = manager.Queue()

    existing_indices = []
    folder = f"data/puzzle_{problem_size}"
    path = f"{folder}/solution_lengths.csv"
    if os.path.exists(path):
        df = pd.read_csv(path).dropna(axis="rows")
        existing_indices = df["problem_index"].values

    num_processes = min(multiprocess.cpu_count() + 1, len(candidate_algorithms) * repetitions + 1, max_processes)
    with multiprocess.Pool(num_processes) as pool:
        pool.apply_async(results_handler,
                         args=(folder, candidate_algorithms, repetitions, queue))

        jobs = []
        for i in range(repetitions):
            print(f"\nStart of problem {i + 1}/{repetitions}")
            position = State.get_random_state(problem_size, exclude_indices=existing_indices).position

            for algorithm in candidate_algorithms:
                job = pool.apply_async(problem_solver, args=(i, position, algorithm, queue))
                jobs.append(job)

        for job in jobs:
            job.get()

        while not queue.empty():
            time.sleep(0.1)
        queue.put((None, (None, None), None, None))

    pool.join()


if __name__ == '__main__':
    main()
