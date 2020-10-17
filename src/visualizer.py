import cv2
import numpy as np

from src.spiral_matrix import SpiralMatrixMapping


class Visualizer(object):

    def __init__(self, image_path, puzzle, scale=1):
        self.n = puzzle.shape[0]
        self.smm = SpiralMatrixMapping(self.n)

        self.image = cv2.imread(image_path)
        self.cells = self._split_image(scale)
        self._permute_cells(puzzle)
        self.show_image = self._cells_to_image()

    def _split_image(self, scale=1):
        """
        Split image to NxN cells
        :param scale: scale of the image
        :return: ndarray with shape (n, n, h, w, c)
        """
        # scale image
        self.image = cv2.resize(self.image, None, fx=scale, fy=scale)
        # resize image
        h = self.image.shape[0] - self.image.shape[0] % self.n
        w = self.image.shape[1] - self.image.shape[1] % self.n
        self.image = cv2.resize(self.image, (w, h))

        H, W, C = self.image.shape
        h, w = H // self.n, W // self.n
        cells = np.empty((self.n, self.n, h, w, C), dtype=self.image.dtype)
        for r in range(self.n):
            for c in range(self.n):
                cell = self.image[h * r:h * r + h, w * c:w * c + w]
                cv2.putText(cell, "%d" % (self.smm.get_spiral_index(r, c) + 1),
                            (0, h - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
                cells[r, c, :] = self.image[h * r:h * r + h, w * c:w * c + w]
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
                if puzzle[i, j] == 0:
                    continue
                r, c = self.smm.get_matrix_indexes(puzzle[i, j] - 1)
                permuted[i, j] = self.cells[r, c]
        self.cells = permuted

    def _cells_to_image(self):
        """
        create image from cells
        :return: ndarray with shape (H, W, C)
        """
        H, W, C = self.image.shape
        h, w = H // self.n, W // self.n
        image = np.empty_like(self.image)
        for r in range(self.n):
            for c in range(self.n):
                image[h * r:h * r + h, w * c:w * c + w] = self.cells[r][c]
        return image

    def show(self, ms=None):
        """
        Show puzzle image
        :param ms: milliseconds for visualization.
        None - will wait until key pressed
        :return: None
        """
        cv2.imshow("n-puzzle image", self.show_image)
        cv2.waitKey(ms)


if __name__ == '__main__':
    from src.parser import parse_n_puzzle

    puzzle = parse_n_puzzle("../data/puzzles/solv_5.txt")
    vis = Visualizer("../data/images/pepega.jpg", puzzle)

    from src.utils.profiler import Profiler
    prof = Profiler("show")

    for i in range(1):
        prof.tick()
        vis.show()
        prof.tock()
    print(prof)

