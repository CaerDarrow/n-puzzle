from heapq import heappop, heappush
import numpy as np
from spiral_matrix import SpiralMatrixMapping


class Field:
    """
        Representation of puzzle field
    """
    def __init__(self, distance_from_start: int = 0, parent=None, state: np.array = None):
        # assert any((distance_from_start, parent))
        self.parent = parent
        if self.parent:
            self.cost = parent.cost + 1
        else:
            self.cost = distance_from_start
        assert state is not None, "Попытка создать пустую головоломку"
        self.state = state

    @property
    def weight(self):
        return self.cost + self.heuristic()

    def heuristic(self):
        # some manipulations with self.state, мб стоит считать это в солвере
        return 0

    def __lt__(self, other):
        return self.weight < other.weight

    def __hash__(self):
        return hash(self.state.tostring())


class Solver:
    def __init__(self, first_field: Field):
        self.closed_set = set()
        self.open_set = []
        self.first_field = first_field
        self.target_state = []  # решение

    def _set_solution(self):
        solution = SpiralMatrixMapping(self.first_field.state.shape[0])
        self.target_state = np.reshape(
            np.roll(
                np.sort(self.first_field.state, axis=None),
                -1
            )[solution.matrix_to_spiral],
            self.first_field.state.shape
        )

    def _is_solvable(self):
        if False:
            return False
        return True

    def _generate_fields(self, field: Field):
        y, x = np.argwhere(field.state == 0)[0]
        y_max, x_max = field.state.shape
        for x_gain, y_gain in (1, 0), (-1, 0), (0, 1), (0, -1):
            if x + x_gain in range(0, x_max) and y + y_gain in range(0, y_max):
                copied = np.copy(field.state)
                copied[y, x], copied[y + y_gain, x + x_gain] = copied[y + y_gain, x + x_gain], copied[y, x]
                if hash(tuple(copied.tostring())) not in self.closed_set:
                    heappush(self.open_set, Field(parent=field, state=copied))
                else:
                    del copied  # lol

    def solve(self):
        if self._is_solvable():
            self._set_solution()
            heappush(self.open_set, self.first_field)
            while self.open_set:
                current_field = heappop(self.open_set)
                print(current_field.state)
                self.closed_set.add(hash(current_field))
                if np.array_equal(current_field.state, self.target_state):
                    print("Solved")
                    return current_field  # solved
                self._generate_fields(field=current_field)  # Fields states generator
        raise Exception("Can't solve")
