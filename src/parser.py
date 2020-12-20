import numpy as np
from pathlib import Path

from .puzzle import Field


def _check_cells(puzzle):
    flatten = np.arange(puzzle.size)
    return (flatten == np.sort(puzzle.flatten())).all()


def parse_n_puzzle(path: str or Path) -> Field:
    """
    Parse file to ndarray
    :param path: str or pathlib.Path. Path to n_puzzel file
    :return: Field
    """
    path = Path(path)

    puzzle: np.ndarray = None

    try:
        text = path.read_text().split('\n')
    except Exception as exc:
        raise Exception("Wrong file format")
    j = 0
    for i, line in enumerate(text):
        try:
            line = line.strip()
            if not line or line[0] == '#':
                continue
            if puzzle is None:
                puzzle = np.zeros((int(line), int(line)), dtype=np.int)
                continue
            separated = line.split()
        except Exception as exc:
            raise Exception("Wrong file content")

        assert len(separated) == puzzle.shape[0], \
            "Puzzle must have %d number or columns" % puzzle.shape[0]

        assert "".join(separated).isdecimal(), \
            "Cell value must be unsigned integer, '+' is prohibited too"
        puzzle[j, :] = separated
        j += 1

    assert j == puzzle.shape[0], "Puzzle must have %d number or rows" % puzzle.shape[0]
    assert _check_cells(puzzle), \
        "Cell value must be unique and 0 <= cell value < n^2 "
    return Field(state=puzzle)


if __name__ == '__main__':

    puzz_dir = Path("../data/puzzles")
    for puzz_path in sorted(puzz_dir.iterdir()):
        print(puzz_path.name)
        print(parse_n_puzzle(puzz_path))
