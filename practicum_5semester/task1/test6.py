from task6 import v1_vector, v2_non_vector, v3_part_vector, gen
import unittest
import numpy as np
import numpy.testing as npt

data = [(np.ones(100), ),
        (np.arange(100), ),
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
