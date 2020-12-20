#!/usr/bin/python3
import argparse
from pathlib import Path
from src.parser import parse_n_puzzle
from src.puzzle import Solver
from src.visualizer import Visualizer


def generator(solution):
    solutions = []
    while solution is not None:
        solutions.append(solution)
        solution = solution.parent
    return list(reversed(solutions))


def parse_arguments(parser):
    parser.add_argument("filename", type=Path, help="generated puzzle file")
    parser.add_argument("-H", "--heuristics", action="append", choices=['Manhattan', 'Linear', 'Hamming'], default=[],
                        help="Add heuristics")
    parser.add_argument("-G", "--greedy_search", action="store_true", default=False,
                        help="Greedy search mode")
    parser.add_argument("-U", "--uniform_cost", action="store_true", default=False,
                        help="Unifrom cost mode")
    parser.add_argument("-D", "--debug", action="store_true", default=False,
                        help="Debug mode")
    parser.add_argument("-V", "--visualize", type=Path, help="Visualization")
    return parser.parse_args()


def main():
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    puzzle = parse_n_puzzle(args.filename)
    solver = Solver(
        first_field=puzzle,
        manhattan='Manhattan' in args.heuristics,
        hamming='Hamming' in args.heuristics,
        linear='Linear' in args.heuristics,
        greedy_search=args.greedy_search,
        uniform_cost=args.uniform_cost,
        debug=args.debug,
    )
    solved_puzzle = generator(solver.solve())
    print("========== Solution ==========")
    for field in solved_puzzle:
        print(field.state)
    print(f"Number of moves: {len(solved_puzzle) - 1}")
    print(f"Time complexity: {len(solver.closed_set)}")
    print(f"Size complexity: {len(solver.closed_set) + len(solver.open_set)}")
    print("===================================")
    if args.visualize:
        vis = Visualizer(args.visualize, puzzle)
        for field in solved_puzzle:
            vis.add_swap(*field.permutations)
        vis.run()


if __name__ == '__main__':
    # python3 main.py data/puzzles/solv_3.txt -H Manhattan -V data/images/pepega.jpg
    try:
        main()
        exit(0)
    except Exception as exc:
        print(exc)
        exit(1)



