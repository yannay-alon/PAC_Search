import torch
import numpy as np
from typing import Optional, Iterator, Sequence, Any

from priority import PriorityQueue, DoublyPriorityQueue
from puzzle import PuzzleProblem, State

from distributions import Distribution
from neural_distribution import NeuralDistribution

STATUS_OPEN = 0
STATUS_CLOSED = 1


class SearchAlgorithm:
    def __init__(self, problem: PuzzleProblem):
        self.problem = problem

    def search(self) -> Optional[tuple[list[State], str, Any]]:
        """
        :return: List of states that define the solution or None if no solution was found
        """
        self.problem.reset()
        for goal, stop, information in self._search():
            solution = [goal]
            while (goal := goal.parent) is not None:
                solution.append(goal)

            solution = solution[::-1]
            yield solution, stop, information
            if stop is not None:
                return

    def _search(self) -> Iterator[tuple[State, Optional[str], Any]]:
        raise NotImplementedError()


class WeightedAStar(SearchAlgorithm):
    def __init__(self, problem: PuzzleProblem, weight: float = 1):
        super().__init__(problem)

        assert 1 <= weight, "Weight must be greater than 1"
        self.weight = weight

    def _search(self) -> Iterator[tuple[State, Optional[str], None]]:
        open_set: PriorityQueue[State, float] = PriorityQueue()
        open_set.add(self.problem.initial_state, self.weight * self.problem.heuristic(self.problem.initial_state))

        while open_set:
            current_state, weighted_f_score = open_set.pop()

            # Check if current state is the goal
            if self.problem.is_solved(current_state):
                yield current_state, "Goal Found", None

            for neighbor in self.problem.expand(current_state):
                # If neighbor is already in open_set, update it using the older version
                if neighbor in open_set:
                    old_neighbor = open_set.get_element(neighbor)
                    if neighbor.depth < old_neighbor.depth:
                        neighbor.update_information(old_neighbor)
                        open_set.update(neighbor, neighbor.depth + self.weight * self.problem.heuristic(neighbor))
                else:
                    # If neighbor is not in open_set, add it
                    open_set.add(neighbor, neighbor.depth + self.weight * self.problem.heuristic(neighbor))
        return None, "Finished Search", None

    def __str__(self):
        if self.weight == 1:
            return "A*"
        return f"{self.weight} Weighted A*"


class AnytimeWeightedAStar(SearchAlgorithm):
    def __init__(self, problem: PuzzleProblem, weight: float = 1,
                 approximation_error: float = 0, required_confidence: float = 1,
                 ratios: Optional[Sequence[float]] = None):
        """

        :param problem: Problem to solve
        :param weight: Weight to apply to the heuristic (f' = g + weight * h)
        :param approximation_error: Maximum error ratio allowed (a.k.a. epsilon)
        :param required_confidence: Required confidence (a.k.a. 1 - delta)
        :param ratios: Ratios of the heuristic values for the initial state h'(s) / h(s)
        """
        super().__init__(problem)

        assert 1 + approximation_error <= weight, "Weight must be greater than 1 + approximation error"

        assert 1 <= weight, "Weight must be greater than 1"
        self.weight = weight

        assert 0 <= approximation_error, "Approximation error must be greater than 0"
        self.approximation_error = approximation_error
        self.approximation_ratio = 1 + approximation_error

        assert 0 <= required_confidence <= 1, "Required confidence must be between 0 and 1"
        self.required_confidence = required_confidence

        if ratios is None:
            self._variables = None
            self._results = None
            self.fast_lb_probability = None
        else:
            ratios = np.array(ratios)
            assert np.all((0 <= ratios) & (ratios <= 1)), "Ratios must be between 0 and 1"

            density, bins = np.histogram(ratios)
            unity_density = density / density.sum()

            self._variables = bins[1:]
            self._results = unity_density.cumsum()

        self.max_f_min = 0

    def cumulative_distribution_function(self, input_value: float) -> float:
        return np.interp(input_value, self._variables, self._results)

    @property
    def parameters(self) -> tuple[float, float, float]:
        return self.weight, self.approximation_error, self.required_confidence

    @property
    def parameters_names(self) -> list[str, str, str]:
        return ["weight", "approximation_error", "required_confidence"]

    def _update_max_f_min(self, node: State, f_value: float) -> bool:
        if self.max_f_min < f_value:
            self.max_f_min = f_value
            return True
        return False

    def _search(self) -> Iterator[tuple[State, Optional[str], list[float]]]:
        open_set: DoublyPriorityQueue[State, float] = DoublyPriorityQueue()
        open_set.add(self.problem.initial_state,
                     self.weight * self.problem.heuristic(self.problem.initial_state),
                     self.problem.heuristic(self.problem.initial_state))

        closed_set: dict[State, State] = dict()

        self.max_f_min = self.problem.heuristic(self.problem.initial_state)
        max_f_min_history = [self.max_f_min]

        solution_score = float("inf")
        solution = None

        while open_set:

            current_state, weighted_f_score, f_score = open_set.pop()

            if solution_score <= f_score:
                continue
            closed_set[current_state] = current_state

            for neighbor in self.problem.expand(current_state):

                if neighbor in open_set:
                    node_status = STATUS_OPEN
                    old_neighbor = open_set.get_element(neighbor)
                    neighbor.update_information(old_neighbor)
                elif neighbor in closed_set:
                    node_status = STATUS_CLOSED
                    old_neighbor = closed_set[neighbor]
                    neighbor.update_information(old_neighbor)
                else:
                    node_status = None
                    old_neighbor = None

                if solution_score <= neighbor.depth + self.problem.heuristic(neighbor):
                    continue

                if self.problem.is_solved(neighbor):
                    solution_score = neighbor.depth
                    solution = neighbor
                    # Note: h*(s) >= U / w  --- This gives another lower bound
                    # self._update_max_f_min(solution, solution_score / self.weight)
                    yield neighbor, self.check_stop_criteria(solution), max_f_min_history
                    continue

                neighbor_weighted_f_score = neighbor.depth + self.weight * self.problem.heuristic(neighbor)
                neighbor_f_score = neighbor.depth + self.problem.heuristic(neighbor)

                if node_status == STATUS_OPEN:
                    if neighbor.depth < old_neighbor.depth:
                        open_set.update(neighbor, neighbor_weighted_f_score, neighbor_f_score)
                    continue

                if node_status == STATUS_CLOSED:
                    if neighbor.depth < old_neighbor.depth:
                        del closed_set[neighbor]
                        open_set.add(neighbor, neighbor_weighted_f_score, neighbor_f_score)
                    continue

                open_set.add(neighbor, neighbor_weighted_f_score, neighbor_f_score)

            if open_set and self._update_max_f_min(*open_set.get_smallest(priority_key=1)):
                max_f_min_history.append(self.max_f_min)
                if solution is not None and (stop_reason := self.check_stop_criteria(solution)) is not None:
                    yield solution, stop_reason, max_f_min_history

        # print("\tFinished Search")
        return solution, "Finished Search", max_f_min_history

    def check_stop_criteria(self, solution: State) -> Optional[str]:
        initial_state_heuristic = self.problem.heuristic(self.problem.initial_state)
        solution_value = solution.depth

        if solution_value <= self.max_f_min * self.approximation_ratio:
            # print(f"\tAchieved Lower Bound: U={solution_value}, LB={self.max_f_min}")
            return "Achieved Lower Bound"

        if self._variables is None:
            return None

        lower = self.cumulative_distribution_function(
            initial_state_heuristic / solution_value)
        upper = self.cumulative_distribution_function(
            initial_state_heuristic * self.approximation_ratio / solution_value)
        bound = self.cumulative_distribution_function(
            initial_state_heuristic / self.max_f_min)
        confidence = (upper - lower) / (bound - lower)

        if self.required_confidence <= confidence:
            # print(f"\tAchieved Confidence: C={confidence:.3f}, 1-delta={self.required_confidence:.2f}")
            return "Achieved Confidence"
        return None

    def __str__(self):
        if self.approximation_error == 0 and self.required_confidence == 1:
            if self.weight == 1:
                return "Anytime A*"
            else:
                return f"Anytime {self.weight}-A*"
        return f"({self.approximation_error:.2f}, {self.required_confidence:.2f}) Anytime {self.weight}-A*"


class ParametricAnytimeWeightedAStar(AnytimeWeightedAStar):
    def __init__(self, problem: PuzzleProblem, weight: float = 1,
                 approximation_error: float = 0, required_confidence: float = 1,
                 ratios: Optional[Sequence[float]] = None,
                 distribution_type: Optional[str] = None):
        """

        :param problem: Problem to solve
        :param weight: Weight to apply to the heuristic (f' = g + weight * h)
        :param approximation_error: Maximum error ratio allowed (a.k.a. epsilon)
        :param required_confidence: Required confidence (a.k.a. 1 - delta)
        :param ratios: Ratios of the heuristic values for the initial state h'(s) / h(s)
        """
        super().__init__(problem, weight, approximation_error, required_confidence, ratios)

        ratios = np.array(ratios)
        self.dist = Distribution(ratios, distribution_type)  # I don't know the bounds of parameters, but it should work
        self.distribution_type = distribution_type

    def cumulative_distribution_function(self, input_value: float) -> float:
        return self.dist.cdf(input_value, self._variables, self._results)

    @property
    def parameters(self) -> tuple[float, float, float, Optional[str]]:
        parameters = super().parameters
        return parameters + (self.distribution_type,)

    @property
    def parameters_names(self) -> list[str, str, str, str]:
        parameters_names = super().parameters_names
        return parameters_names + ["distribution_type"]

    def __str__(self):
        return f"{super().__str__()} ({self.distribution_type})"


class NeuralAnytimeWeightedAStar(AnytimeWeightedAStar):
    def __init__(self, problem: PuzzleProblem, model: NeuralDistribution,
                 weight: float = 1, approximation_error: float = 0, required_confidence: float = 1):
        super().__init__(problem, weight, approximation_error, required_confidence)

        self.model = model
        self.model.eval()

    def cumulative_distribution_function(self, target_value: float,
                                         heuristic_value: float, lower_bound: float) -> float:
        tensor = torch.tensor([target_value, heuristic_value, lower_bound], dtype=torch.float32)

        cdf, log_prob = self.model.prob(*tensor)
        return cdf.item()

    def check_stop_criteria(self, solution: State) -> Optional[str]:
        initial_state_heuristic = self.problem.heuristic(self.problem.initial_state)
        solution_value = solution.depth

        if solution_value <= self.max_f_min * self.approximation_ratio:
            # print(f"\tAchieved Lower Bound: U={solution_value}, LB={self.max_f_min}")
            return "Achieved Lower Bound"

        # Search-aware PAC condition (with positive bound, i.e. using 1 - \delta directly)
        lower = self.cumulative_distribution_function(solution_value, initial_state_heuristic, self.max_f_min)
        upper = self.cumulative_distribution_function(solution_value / self.approximation_ratio,
                                                      initial_state_heuristic, self.max_f_min)
        bound = self.cumulative_distribution_function(self.max_f_min, initial_state_heuristic, self.max_f_min)
        confidence = (upper - lower) / (bound - lower)

        if self.required_confidence <= confidence:
            # print(f"\tAchieved Confidence: C={confidence:.3f}, 1-delta={self.required_confidence:.2f}")
            return "Achieved Confidence"
        return None

    @property
    def parameters(self) -> tuple[float, float, float, str]:
        parameters = super().parameters
        return parameters + ("Neural",)

    @property
    def parameters_names(self) -> list[str, str, str, str]:
        parameters_names = super().parameters_names
        return parameters_names + ["distribution_type"]
