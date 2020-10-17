import numpy as np

class SpiralMatrixMapping(object):

    def __init__(self, n):
        """
        :param n: side length of a square matrix
        """
        self.n = n
        self.spiral_to_matrix = self._create_spiral_to_matrix()
        self.matrix_to_spiral = self._create_matrix_to_spiral()

    def get_matrix_indexes(self, i) -> np.ndarray:
        """
        :param i: spiral index
        :return: [r, c] column indexes
        """
        return self.spiral_to_matrix[i]

    def get_spiral_index(self, r, c) -> int:
        """
        :param r: row index
        :param c: column index
        :return: spiral index
        """
        return self.matrix_to_spiral[r, c]

    def _create_spiral_to_matrix(self) -> np.ndarray:
        """
        :param n: side length of a square matrix
        :return: nd array with shape (n^2, 2)
                Each row is [x, y] of spiral index in matrix
        """
        k = 0   # starting row index
        l = 0   # starting column index

        m = n = self.n

        indexes = []

        while k < m and l < n:

            # First row from the remaining rows
            for i in range(l, n):
                indexes.append([k, i])

            k += 1

            # Last column from the remaining columns
            for i in range(k, m):
                indexes.append([i, n - 1])

            n -= 1

            # Last row from the remaining rows
            if k < m:
                for i in range(n - 1, (l - 1), -1):
                    indexes.append([m - 1, i])
                m -= 1

            # First column from the remaining columns
            if l < n:
                for i in range(m - 1, k - 1, -1):
                    indexes.append([i, l])
                l += 1
        return np.array(indexes)

    def _create_matrix_to_spiral(self) -> np.ndarray:
        """
        :return: ndarray matrix with shape (self.n, self.n)
        """
        matrix_to_spiral = np.zeros((self.n, self.n), dtype=np.int)
        i = 0
        for y, x in self.spiral_to_matrix:
            matrix_to_spiral[y, x] = i
            i += 1
        return matrix_to_spiral


if __name__ == '__main__':
    n = 4
    smm = SpiralMatrixMapping(n)

    print(smm.matrix_to_spiral)
    print(smm.get_matrix_indexes(13))
    print(smm.get_spiral_index(2, 1))
