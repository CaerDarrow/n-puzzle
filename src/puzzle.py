from heapq import heappop, heappush


class Field:
    """
        Representation of puzzle field
    """
    def __init__(self, distance_from_start=0, parent=None):
        # assert any((distance_from_start, parent))
        self.parent = parent
        if self.parent:
            self.cost = parent.cost + 1
        else:
            self.cost = distance_from_start

        self.state = [0, 1, 2, 4, 5, 6, 7, 8]

    @property
    def weight(self):
        return self.cost + self.heuristic()

    def heuristic(self):
        # some manipulations with self.state, мб стоит считать это в солвере
        return 0

    def __lt__(self, other):
        return self.weight < other.weight

    def __hash__(self):
        return hash(self.state)


class Solver:
    def __init__(self, first_field: Field):
        self.closed_set = set()
        self.open_set = []
        self.first_field = first_field
        self.target_state = []  # решение

    def _is_solvable(self):
        # check and return False if not, else:
        heappush(self.open_set, self.first_field)
        return True

    def solve(self):
        if self._is_solvable():
            while self.open_set:
                current_field = heappop(self.open_set)
                if current_field not in self.closed_set:
                    self.closed_set.add(current_field)
                    if current_field.state == self.target_state:
                        return True  # solved
                    for field in [Field(parent=current_field), ]:  # Fields states generator
                        heappush(self.open_set, field)

        raise Exception("Can't solve")
