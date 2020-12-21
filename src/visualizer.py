import cv2
import sys
import numpy as np
from pathlib import Path
from copy import deepcopy

from .puzzle import Field
from .spiral_matrix import SpiralMatrixMapping


class Visualizer(object):

    def __init__(self, image_path: str or Path, puzzle: Field, scale=1.0):
        """
        :param image_path: str ot Path - path to image for n-puzzle visualization
        :param puzzle: Field object
        :param scale: float - image scale
        """
        self.n = puzzle.state.shape[0]
        self.smm = SpiralMatrixMapping(self.n)

        self.state = np.copy(puzzle.state)
        self.target = np.zeros((self.n, self.n), dtype=np.int)
        self.target[tuple(self.smm.spiral_to_matrix.T)] = np.roll(np.arange(self.n**2), -1)

        self.zero_xy = np.where(puzzle.state == 0)
        self.zero_xy = [self.zero_xy[0][0], self.zero_xy[1][0]]
        self.next_xy = deepcopy(self.zero_xy)
        self.swap_history = []

        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise AssertionError("Wrong image path")
        self.cell_h = 0
        self.cell_w = 0

        self.cells = self._split_image(scale)
        self._permute_cells(puzzle.state)
        self.puzzle_image = self._cells_to_image()

    def add_swap(self, xy1, xy2):
        """
        :param xy1: (row, column)
        :param xy2: (row, column)
        :return: None
        """
        xy1, xy2 = tuple(xy1), tuple(xy2)
        self.swap_history.append((xy1, xy2))

    def _check_state(self):
        return (self.state == self.target).all()

    def run(self):
        """
        Run visualization
        """
        def _print_progress():
            progress = "Progress: {:-{}d} / {:-{}d} - {}% " \
                .format(i, num_spaces, history_len, num_spaces, int(i / history_len * 100))
            sys.stdout.write("\r" + progress)

        print("========== Visualization ==========")
        print("Enter        - one step forward")
        print("Backspace    - one step backward")
        print("Space        - run / stop")
        print("Esc          - exit")
        print("===================================")

        history_len = len(self.swap_history)
        num_spaces = len(str(history_len))
        run = False
        i = 0
        ms = 0
        while True:
            _print_progress()
            image = self.image if self._check_state() else self.puzzle_image
            key = self.show(image, ms)
            if key == ord(' '):
                run = ~run
            elif key == 13 and i < history_len:     # FORWARD
                self._swap_cells(*self.swap_history[i])
                i += 1
                ms = 0
                run = False
            elif key == 8 and i != 0:   # BACKWARD
                i -= 1
                self._swap_cells(*self.swap_history[i])
                ms = 0
                run = False
            elif key == 27:     # EXIT
                #
                print()
                cv2.destroyWindow("%d-puzzle" % self.n)
                break

            # RUN / STOP
            if (key == ord(' ') or key == -1) and run and i < history_len:
                self._swap_cells(*self.swap_history[i])
                ms = 250
                i += 1
            elif key == ord(' ') and not run:
                ms = 0

    def manual_control(self):
        """
        Run manual visualization
        """
        print("========== Visualization ==========")
        print("W    - swap top")
        print("A    - swap left")
        print("S    - swap bottom")
        print("D    - swap right")
        print("===================================")

        def perform_step(axis, step):
            new_val = self.next_xy[axis] + step
            if self.n - 1 < new_val or new_val < 0:
                return
            self.next_xy[axis] = new_val
            self._swap_cells(tuple(self.zero_xy), tuple(self.next_xy))
            self.zero_xy = deepcopy(self.next_xy)

        while True:
            image = self.image if self._check_state() else self.puzzle_image
            key = self.show(image, 0)
            if key == ord('w'):     # TOP
                perform_step(0, -1)
            elif key == ord('a'):   # LEFT
                perform_step(1, -1)
            elif key == ord('s'):   # BOTTOM
                perform_step(0, 1)
            elif key == ord('d'):   # RIGHT
                perform_step(1, 1)
            elif key == 27:         # EXIT
                cv2.destroyWindow("%d-puzzle" % self.n)
                break

    def show(self, image, ms=0):
        """
        Show puzzle image
        :param ms: milliseconds for visualization.
                    0 - will wait until key pressed
        :return: int - key
        """
        cv2.imshow("%d-puzzle" % self.n, image)
        key = cv2.waitKey(ms)
        return key

    def _swap_cells(self, xy1, xy2):
        """
        :param xy1: (row, column)
        :param xy2: (row, column)
        :return: None
        """
        self.state[xy1], self.state[xy2] = self.state[xy2], self.state[xy1]
        cell1 = np.copy(self.cells[xy1])
        cell2 = self.cells[xy2]

        c_lt, c_rb = self._cell_to_image_indexes(*xy2)
        self.puzzle_image[c_lt[0]:c_lt[1], c_rb[0]:c_rb[1]] = cell1
        c_lt, c_rb = self._cell_to_image_indexes(*xy1)
        self.puzzle_image[c_lt[0]:c_lt[1], c_rb[0]:c_rb[1]] = cell2

        self.cells[xy1] = cell2
        self.cells[xy2] = cell1

    def _cell_to_image_indexes(self, r, c):
        """
        :return: (left top), (right bottom)
        """
        return (self.cell_h * r, self.cell_h * r + self.cell_h),\
               (self.cell_w * c, self.cell_w * c + self.cell_w)

    @staticmethod
    def _fake_rgba(image, alpha=25):
        return image * (alpha / 255)

    def _split_image(self, scale=1):
        """
        Split image to NxN cells
        :param scale: scale of the image
        :return: ndarray with shape (n, n, h, w, c)
        """
        # scale image
        h, w, _ = self.image.shape
        if h * scale < 3 or w * scale < 3:
            raise AssertionError("Image resolution too low")
        if h * scale > 5000 or w * scale > 5000:
            raise AssertionError("Image resolution too high")
        self.image = cv2.resize(self.image, None, fx=scale, fy=scale)
        # resize image
        h = self.image.shape[0] - self.image.shape[0] % self.n
        w = self.image.shape[1] - self.image.shape[1] % self.n
        self.image = cv2.resize(self.image, (w, h))

        H, W, C = self.image.shape
        h, w = H // self.n, W // self.n
        self.cell_h = h
        self.cell_w = w
        cells = np.empty((self.n, self.n, h, w, C), dtype=self.image.dtype)
        for r in range(self.n):
            for c in range(self.n):
                c_lt, c_rb = self._cell_to_image_indexes(r, c)
                cell = np.copy(self.image[c_lt[0]:c_lt[1], c_rb[0]:c_rb[1]])
                spiral_ind = self.smm.get_spiral_index(r, c) + 1
                if spiral_ind == self.n ** 2:
                    spiral_ind = 0
                    cell = self._fake_rgba(cell)
                cv2.putText(cell, "%d" % spiral_ind,
                            (0, h - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
                cells[r, c, :] = cell
        return cells

    def _permute_cells(self, puzzle):
        """
        Permute cells according to puzzle
        :param puzzle: ndarray of puzzle with shape (n, n)
        :return: None
        """
        permuted = np.empty_like(self.cells)
        for i in range(self.cells.shape[0]):
            for j in range(self.cells.shape[0]):
                spiral_ind = puzzle[i, j]
                if spiral_ind == 0:
                    spiral_ind = self.n ** 2
                r, c = self.smm.get_matrix_indexes(spiral_ind - 1)
                permuted[i, j] = self.cells[r, c]
        self.cells = permuted

    def _cells_to_image(self):
        """
        Create image from cells
        :return: ndarray with shape (H, W, C)
        """
        image = np.empty_like(self.image)
        for r in range(self.n):
            for c in range(self.n):
                c_lt, c_rb = self._cell_to_image_indexes(r, c)
                image[c_lt[0]:c_lt[1], c_rb[0]:c_rb[1]] = self.cells[r][c]
        return image


if __name__ == '__main__':

    from parser import parse_n_puzzle

    puzzle = parse_n_puzzle("../data/puzzles/solv_3.txt")

    vis = Visualizer("../data/images/pepega.jpg", puzzle)
    vis.add_swap([0, 2], [0, 1])
    vis.add_swap(np.array((0, 1)), np.array((1, 1)))
    vis.add_swap((1, 1), (2, 1))
    vis.add_swap((2, 1), (2, 0))
    vis.add_swap((2, 0), (1, 0))
    vis.add_swap((1, 0), (0, 0))
    # vis.run()
    vis.manual_run()
