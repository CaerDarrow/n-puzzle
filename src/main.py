from pathlib import Path
from parser import parse_n_puzzle
from puzzle import Solver, Field


def generator(solution):
    solutions = []
    while solution is not None:
        solutions.append(solution)
        solution = solution.parent
    return reversed(solutions)


def main(puzzle_name):
    puzzle_path = Path(f"../data/puzzles/{puzzle_name}")
    puzzle = parse_n_puzzle(puzzle_path)
    solver = Solver(first_field=puzzle)
    solved_puzzle = solver.solve()
    for field in generator(solved_puzzle):
        print(field.state)
        print("===")


if __name__ == '__main__':
    # import sys
    # puzzle_name = sys.argv[1]
    puzzle_name = 'solv_3.txt'
    main(puzzle_name)



