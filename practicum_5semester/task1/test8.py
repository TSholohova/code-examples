from task8 import v1_vector, v2_non_vector, gen
from task8 import v3_part_vector as v3_scipy
import unittest
import numpy as np
import numpy.testing as npt

data = [(np.arange(16).reshape(4, 4),
         np.arange(4), np.eye(4)),
        (np.array([[17]]), np.array([0]), np.array([[1]])),
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
