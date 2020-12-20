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
                 distance_from_start: int = 0,
                 manhattan_score: int = 0,
                 hamming_score: int = 0,
                 linear_conflict: int = 0,
                 permutations=((0, 0), (0, 0))):
        self.parent = parent
        # g(n)
        self.exact_cost = distance_from_start
        # h(n)
        self.manhattan_score = manhattan_score
        self.hamming_score = hamming_score
        self.linear_conflict = linear_conflict
        # state
        assert state is not None, "Попытка создать пустую головоломку"
        self.state = state
        self.permutations = permutations

    @property
    def cost(self):  # f(n)
        return self.exact_cost + self.estimated_cost

    @property
    def estimated_cost(self):  # h(n) - heuristic
        return self.manhattan_score + self.hamming_score + self.linear_conflict

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
    """
        Solver
    """
    def __init__(
            self,
            first_field: Field,
            greedy_search: bool = False,
            uniform_cost: bool = False,
            manhattan: bool = False,
            hamming: bool = False,
            linear: bool = False,
    ):
        self.closed_set = set()
        self.open_set = []
        self._first_field = first_field
        self._target_state = np.ndarray([])  # решение
        self._field_size = self._first_field.state.shape[0]
        self._field_range = tuple(range(self._first_field.state.shape[0]))
        self._greedy = greedy_search
        self._uniform_cost = uniform_cost
        self._manhattan = manhattan
        self._hamming = hamming
        self._linear = linear
        if self._uniform_cost and any((self._manhattan, self._hamming, self._linear)):
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
        for y in self._field_range:
            for x in self._field_range:
                if state[y, x] != 0:
                    target_y, target_x = np.argwhere(self.target_state == state[y, x])[0]
                    distance += abs(target_y - y) + abs(target_x - x)
        return distance

    def _hamming_score(self, state: np.ndarray) -> int:
        return np.count_nonzero(state != self.target_state)

    def _linear_conflict(self, state: np.ndarray) -> int:
        conflicts = 0
        for y in self._field_range:
            for x in self._field_range:
                current = state[y, x]
                current_target_y, current_target_x = np.argwhere(self.target_state == current)[0]
                if current != 0:
                    for test_y in range(y + 1, self._field_size):
                        test = state[test_y, x]
                        if test != 0:
                            test_target_y, test_target_x = np.argwhere(self.target_state == test)[0]
                            if test_target_x == current_target_x and test_target_y < current_target_y and test_y > y:
                                conflicts += 1
                    for test_x in range(x + 1, self._field_size):
                        test = state[y, test_x]
                        if test != 0:
                            test_target_y, test_target_x = np.argwhere(self.target_state == state[y, test_x])[0]
                            if test_target_y == current_target_y and test_target_x < current_target_x and test_x > x:
                                conflicts += 1

        return conflicts * 2

    def _count_inversions(self):
        flat_solution = self.target_state.flatten()
        flat_start = self._first_field.state.flatten()
        inversions = 0
        for i, val1 in enumerate(flat_start):
            pos1 = np.argwhere(flat_solution == val1)[0][0]
            for j, val2 in enumerate(flat_start[i + 1:], start=i + 1):
                pos2 = np.argwhere(flat_solution == val2)[0][0]
                if pos2 < pos1:
                    inversions += 1
        return inversions

    def _is_solvable(self) -> bool:
        inversions = self._count_inversions()
        z_start = np.argwhere(self._first_field.state == 0)[0]
        z_solution = np.argwhere(self.target_state == 0)[0]
        zero_distance = abs(z_start[0] - z_solution[0]) + abs(z_start[1] - - z_solution[1])
        if zero_distance % 2 == 0 and inversions % 2 == 0:
            return True
        if zero_distance % 2 == 1 and inversions % 2 == 1:
            return True
        return False

    def _generate_fields(
            self,
            field: Field,
    ):
        y, x = np.argwhere(field.state == 0)[0]
        for x_gain, y_gain in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (x_gain != 0 and x + x_gain in self._field_range) or (y_gain != 0 and y + y_gain in self._field_range):
                copied = np.copy(field.state)
                copied[y, x], copied[y + y_gain, x + x_gain] = copied[y + y_gain, x + x_gain], copied[y, x]
                if hash(copied.tostring()) not in self.closed_set:
                    heappush(
                        self.open_set,
                        Field(
                            parent=field,
                            state=copied,
                            distance_from_start=field.exact_cost + 1 if not self._greedy else 0,
                            manhattan_score=self._manhattan_score(copied) if self._manhattan else 0,
                            hamming_score=self._hamming_score(copied) if self._hamming else 0,
                            linear_conflict=self._linear_conflict(copied) if self._linear else 0,
                            permutations=((y, x), (y + y_gain, x + x_gain))
                        )
                    )

    def solve(self) -> Field:
        self._set_solution()
        if self._is_solvable():
            heappush(self.open_set, self._first_field)
            while self.open_set:
                current_field = heappop(self.open_set)
                if current_field.parent:
                    print(f"{current_field.cost} = {current_field.exact_cost} + {current_field.estimated_cost}")
                # print(current_field.state)
                self.closed_set.add(hash(current_field))
                if np.array_equal(current_field.state, self.target_state):
                    return current_field  # solved
                self._generate_fields(field=current_field)  # Fields states generator
        raise Exception("Can't solve")
