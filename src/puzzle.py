from heapq import heappop, heappush
import numpy as np
from spiral_matrix import SpiralMatrixMapping


class Field:
    """
        Representation of puzzle field
    """
    def __init__(self,
                 parent=None,
                 state: np.array = None,
                 distance_from_start: bool = True,
                 manhattan_score: int = 0,
                 hamming_score: int = 0,
                 permutations=((0, 0), (0, 0))):
        self.parent = parent
        # g(n)
        self.exact_cost = 0
        if distance_from_start and parent:
            self.exact_cost = parent.exact_cost + 1
        # h(n)
        self.manhattan_score = manhattan_score
        self.hamming_score = hamming_score
        # state
        assert state is not None, "Попытка создать пустую головоломку"
        self.state = state
        self.permutations = permutations

    @property
    def cost(self):  # f(n)
        return self.exact_cost + self.estimated_cost

    @property
    def estimated_cost(self):  # h(n) - heuristic
        return self.manhattan_score + self.hamming_score

    def __lt__(self, other):
        return self.cost < other.cost

    def __le__(self, other):
        return self.cost <= other.cost

    def __gt__(self, other):
        return self.cost > other.cost

    def __ge__(self, other):
        return self.cost >= other.cost

    def __hash__(self):
        return hash(self.state.tostring())


class Solver:
    def __init__(
            self,
            first_field: Field,
            greedy_search: bool = False,
            uniform_cost: bool = False,
            manhattan: bool = False,
            hamming: bool = False,
    ):
        self.closed_set = set()
        self.open_set = []
        self._first_field = first_field
        self._target_state = np.ndarray([])  # решение
        self._field_size = tuple(range(self._first_field.state.shape[0]))
        self._greedy = greedy_search
        self._uniform_cost = uniform_cost
        self._manhattan = manhattan
        self._hamming = hamming
        if self._uniform_cost and any((self._manhattan, self._hamming)):
            raise AssertionError("Нельзя использовать эвристики и унифицированную стоимость")

    def _set_solution(self):
        solution = SpiralMatrixMapping(self._first_field.state.shape[0])
        self.target_state = np.reshape(
            np.roll(
                np.sort(self._first_field.state, axis=None),
                -1
            )[solution.matrix_to_spiral],
            self._first_field.state.shape
        )

    def _manhattan_score(self, state: np.ndarray) -> int:
        distance = 0
        for y in self._field_size:
            for x in self._field_size:
                if state[y, x] != 0:
                    target_y, target_x = np.argwhere(self.target_state == state[y, x])[0]
                    distance += abs(target_y - y) + abs(target_x - x)
        return distance

    def _hamming_score(self, state: np.ndarray) -> int:
        return np.count_nonzero(state != self.target_state)

    def _is_solvable(self) -> bool:
        if False:
            return False
        return True

    def _generate_fields(
            self,
            field: Field,
    ):
        y, x = np.argwhere(field.state == 0)[0]
        for x_gain, y_gain in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (x_gain != 0 and x + x_gain in self._field_size) or (y_gain != 0 and y + y_gain in self._field_size):
                copied = np.copy(field.state)
                copied[y, x], copied[y + y_gain, x + x_gain] = copied[y + y_gain, x + x_gain], copied[y, x]
                if hash(copied.tostring()) not in self.closed_set:
                    manhattan_score = 0
                    hamming_score = 0
                    distance_from_start = not self._greedy
                    if not self._uniform_cost:
                        if self._manhattan:
                            manhattan_score = self._manhattan_score(copied)
                        if self._hamming:
                            hamming_score = self._hamming_score(copied)
                    heappush(
                        self.open_set,
                        Field(
                            parent=field,
                            state=copied,
                            distance_from_start=distance_from_start,
                            manhattan_score=manhattan_score,
                            hamming_score=hamming_score,
                            permutations=((y, x), (y + y_gain, x + x_gain))
                        )
                    )

    def solve(self) -> Field:
        if self._is_solvable():
            self._set_solution()
            heappush(self.open_set, self._first_field)
            while self.open_set:
                current_field = heappop(self.open_set)
                if current_field.parent:
                    print(current_field.exact_cost, current_field.estimated_cost, current_field.parent.estimated_cost)
                # print(current_field.state)
                self.closed_set.add(hash(current_field))
                if np.array_equal(current_field.state, self.target_state):
                    return current_field  # solved
                self._generate_fields(field=current_field)  # Fields states generator
        raise Exception("Can't solve")
