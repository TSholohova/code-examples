from task3 import v1_vector, v2_non_vector, v3_part_vector, gen
import unittest
import numpy as np
import numpy.testing as npt

data = [(np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
         np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])),
        (np.array([0, 0, 0, 0]),
         np.array([0, 0, 0, 0])),
        gen(3),
        gen(4)]


class TestTask1(unittest.TestCase):

    def test_v1_vector(self):
        for d in data:
            npt.assert_almost_equal(v1_vector(*d), v2_non_vector(*d))

    def test_v3_part_vector(self):
        for d in data:
            npt.assert_almost_equal(v3_part_vector(*d), v2_non_vector(*d))

if __name__ == "__main__":
    unittest.main()
