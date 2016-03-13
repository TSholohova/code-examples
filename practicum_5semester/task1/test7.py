from task7 import v1_vector, v2_non_vector, gen
from task7 import v3_part_vector as v3_scipy
import unittest
import numpy as np
import numpy.testing as npt

data = [(np.ones((10, 20)), np.zeros((30, 20))),
        (np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0],
                   [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]]),
        np.array([[10, 10, 10], [20, 20, 20], [30, 30, 30]])),
        gen(3),
        gen(4)]


class TestTask1(unittest.TestCase):

    def test_v1_vector(self):
        for d in data:
            npt.assert_allclose(v1_vector(*d), v3_scipy(*d))

    def test_v2_non_vector(self):
        for d in data:
            npt.assert_allclose(v2_non_vector(*d), v3_scipy(*d))

if __name__ == "__main__":
    unittest.main()
