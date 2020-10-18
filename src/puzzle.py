from heapq import heappop, heappush
import numpy as np
from scipy.spatial.distance import cdist
from spiral_matrix import SpiralMatrixMapping


class Field:
    """
        Representation of puzzle field
    """
    def __init__(self,
                 distance_from_start: int = 0,
                 parent=None,
                 state: np.array = None,
                 manhattan_score: int = 0,
                 perms=((0, 0), (0, 0))):
        # assert any((distance_from_start, parent))
        self.parent = parent
        if self.parent:
            self.exact_cost = parent.exact_cost + 1  # g(n)
        else:
            self.exact_cost = distance_from_start  # g(n)
        assert state is not None, "Попытка создать пустую головоломку"
        self.state = state
        self.manhattan_score = manhattan_score
        self.permutations = perms

    @property
    def cost(self):  # f(n)
        return self.exact_cost + self.estimated_cost()

    def estimated_cost(self):  # # h(n) - heuristic
        return self.manhattan_score

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
    def __init__(self, first_field: Field):
        self.closed_set = set()
        self.open_set = []
        self.first_field = first_field
        self.target_state = np.ndarray([])  # решение
        self.field_size = tuple(range(self.first_field.state.shape[0]))

    def _set_solution(self):
        solution = SpiralMatrixMapping(self.first_field.state.shape[0])
        self.target_state = np.reshape(
            np.roll(
                np.sort(self.first_field.state, axis=None),
                -1
            )[solution.matrix_to_spiral],
            self.first_field.state.shape
        )

    def manhattan_score(self, state):
        distance = 0
        for y in self.field_size:
            for x in self.field_size:
                if state[y, x] != 0:
                    target_y, target_x = np.argwhere(self.target_state == state[y, x])[0]
                    distance += abs(target_y - y) + abs(target_x - x)
        return distance

    def _is_solvable(self):
        if False:
            return False
        return True

    def _generate_fields(self, field: Field):
        y, x = np.argwhere(field.state == 0)[0]
        for x_gain, y_gain in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if x + x_gain in self.field_size and y + y_gain in self.field_size:
                copied = np.copy(field.state)
                copied[y, x], copied[y + y_gain, x + x_gain] = copied[y + y_gain, x + x_gain], copied[y, x]
                if hash(tuple(copied.tostring())) not in self.closed_set:
                    heappush(
                        self.open_set,
                        Field(
                            parent=field,
                            state=copied,
                            manhattan_score=self.manhattan_score(copied),
                            perms=((y, x), (y + y_gain, x + x_gain))
                        )
                    )
                else:
                    del copied  # lol

    def solve(self):
        if self._is_solvable():
            self._set_solution()
            heappush(self.open_set, self.first_field)
            while self.open_set:
                current_field = heappop(self.open_set)
                self.closed_set.add(hash(current_field))
                if np.array_equal(current_field.state, self.target_state):
                    print("Solved")
                    return current_field  # solved
                self._generate_fields(field=current_field)  # Fields states generator
        raise Exception("Can't solve")
