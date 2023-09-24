# import numba as nb
import numpy as np
from typing import Optional, Union, Iterator, Sequence

BLANK = 0  # Must be 0! (not generic)
NEW_AXIS = np.newaxis

MOVEMENT_DIRECTIONS = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])


class State:
    def __init__(self, position: np.ndarray | list, parent: Optional["State"] = None,
                 blank_position: Optional[np.ndarray] = None):
        if isinstance(position, list):
            position = np.array(position)

        self.position = position
        height, width = self.position.shape
        self.size = height

        self.id = hash(tuple(self.position.flatten()))
        if blank_position is not None:
            self.blank_index = blank_position
        else:
            self.blank_index = np.argwhere(self.position == BLANK)[0]

        self.parent = parent

        if parent is None:
            self.depth = float("inf")
        else:
            self.depth = parent.depth + 1

        self.heuristic_value = None
        self.possible_moves = None

    def get_possible_moves(self) -> tuple[list["State"], bool]:
        has_expanded = False
        if self.possible_moves is None:
            self.possible_moves = [(position, location) for position, location in self.get_possible_positions()]
            has_expanded = True

        return [State(position, self, location) for position, location in self.possible_moves
                if self.parent is None or not np.array_equal(position, self.parent.position)], has_expanded

    def get_possible_positions(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        new_locations = self.blank_index + MOVEMENT_DIRECTIONS
        legal_mask = np.all((0 <= new_locations) & (new_locations < self.size), axis=1)

        source = tuple(self.blank_index)

        for new_location in new_locations[legal_mask]:
            new_position = self.position.copy()
            target_y, target_x = new_location

            new_position[source] = new_position[target_y, target_x]
            new_position[target_y, target_x] = BLANK

            yield new_position, new_location

    def is_solvable(self):
        flat_array = self.position.flatten()

        blank_index = np.where(flat_array == BLANK)[0].item()
        inversions = np.sum(np.tril(flat_array > flat_array.reshape(-1, 1))) - blank_index

        if self.size % 2 == 1:
            return inversions % 2 == 0

        row_count = self.size - blank_index // self.size
        return (inversions + row_count) % 2 == 1

    def count_misplaced(self, target_position: np.ndarray):
        return np.count_nonzero(self.position.flatten() != target_position)

    def calculate_manhattan_distance(self, target_position: np.ndarray) -> float:
        index = self.blank_index[0] * self.size + self.blank_index[1]  # Ravel the index

        source_position = self.position[..., NEW_AXIS, NEW_AXIS]
        target_position = target_position[NEW_AXIS, NEW_AXIS]

        locations = np.argwhere(target_position == source_position)
        locations[index] = 0

        source, target = locations[:, :2], locations[:, 2:]
        return np.sum(np.abs(source - target)).item()

    # def calculate_manhattan_distance(self, target_position: np.ndarray):
    #     index = self.blank_index[0] * self.size + self.blank_index[1]  # Ravel the index
    #
    #     source_position = self.position[..., NEW_AXIS, NEW_AXIS]
    #     target_position = target_position[NEW_AXIS, NEW_AXIS]
    #     return self._calculate_manhattan_distance(source_position, target_position, index)
    #
    # @staticmethod
    # @nb.njit()
    # def _calculate_manhattan_distance(source_position: np.ndarray, target_position: np.ndarray, index: int):
    #     locations = np.argwhere(target_position == source_position)
    #     locations[index] = 0
    #
    #     source, target = locations[:, :2], locations[:, 2:]
    #     return np.sum(np.abs(source - target))

    def heuristic(self, target_position: np.ndarray):
        if self.heuristic_value is not None:
            return self.heuristic_value

        heuristic_value = self.calculate_manhattan_distance(target_position)
        self.heuristic_value = heuristic_value
        return heuristic_value

    def update_information(self, other: "State"):
        """
        Update costly information from another state (only invariant information)
        """
        self.heuristic_value = other.heuristic_value

        self.possible_moves = other.possible_moves

    def __eq__(self, other: Union["State", np.ndarray]):
        if isinstance(other, np.ndarray):
            return np.array_equal(self.position, other)
        return np.array_equal(self.position, other.position)

    def __ge__(self, other: "State"):
        return self.depth >= other.depth

    def __gt__(self, other: "State"):
        return self.depth > other.depth

    def __le__(self, other: "State"):
        return self.depth <= other.depth

    def __lt__(self, other: "State"):
        return self.depth < other.depth

    def __hash__(self):
        return self.id

    @property
    def index(self) -> str:
        return ",".join(map(str, self.position.flatten()))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        cell_width = max(len(str(value)) for value in self.position.flatten())
        cell_width = (2 + cell_width % 2) * cell_width
        string_rows = ["┌" + "┬".join(["─" * cell_width] * self.size) + "┐"]
        for row in self.position:
            new_string_row = ["│"]
            for value in row:
                if value == BLANK:
                    new_string_row.append(" " * cell_width)
                else:
                    pad_size = cell_width - len(str(value))
                    pad_right = pad_size // 2
                    pad_left = pad_size - pad_right
                    new_string_row.append(" " * pad_left + str(value) + " " * pad_right)
                new_string_row.append("│")
            string_rows.append("".join(new_string_row))
            string_rows.append("├" + "┼".join(["─" * cell_width] * self.size) + "┤")
        string_rows[-1] = "└" + "┴".join(["─" * cell_width] * self.size) + "┘"
        string_rows = ["", *string_rows, ""]
        return "\n".join(string_rows)

    @staticmethod
    def get_random_state(size: int, exclude_indices: Optional[Sequence[str]] = None) -> "State":
        if exclude_indices is None:
            exclude_indices = []

        while True:
            position = np.random.permutation(size * size).reshape(size, size)
            state = State(position)
            if state.index not in exclude_indices and state.is_solvable():
                return state


class PuzzleProblem:
    def __init__(self, *, initial_state: Optional[State] = None, size: Optional[int] = None):
        if initial_state is None:
            assert size is not None, "Cannot create a problem without initial_state nor size"
            initial_state = State.get_random_state(size)
        else:
            assert size is None or initial_state.size == size, \
                "Cannot have both initial_state and size defined with incompatible values"
            assert initial_state.is_solvable(), "The given initial state is not solvable"

        self.size = initial_state.size
        self.initial_state = initial_state

        self.expanded_nodes = None
        self.reset()

        self.target_position = np.roll(np.arange(self.size * self.size), -1).reshape(self.size, self.size)

    def is_solved(self, state: State):
        return state == self.target_position

    def reset(self, new_initial_state: Optional[State] = None):
        if new_initial_state is not None:
            self.initial_state = new_initial_state
        self.initial_state.depth = 0
        self.initial_state.heuristic_value = None
        self.expanded_nodes = 0

    def heuristic(self, state: State):
        return state.heuristic(self.target_position)

    def expand(self, state: State):
        moves, new_expand = state.get_possible_moves()
        if new_expand:
            self.expanded_nodes += 1
        return moves

    @property
    def index(self) -> str:
        return self.initial_state.index
