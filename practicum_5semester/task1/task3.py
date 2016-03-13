import numpy as np


# v1_vector(x, y : 1d np.arrays)
# returns (True or False)
def v1_vector(x, y):
    return np.all(np.sort(x) == np.sort(y))


# v2_non_vector(x, y : 1d np.arrays)
# returns (True or False)
def v2_non_vector(x, y):
    return sorted(x) == sorted(y)


# v3_part_vector(x, y : 1d np.arrays)
# returns (True or False)
def v3_part_vector(x, y):
    z = np.sort(x) == np.sort(y)
    res = True
    for i in z:
        res = res and i
    return res


# gen(0 <= size <= 4: int) return tuple(x, y : 1d np.arrays of int, )
def gen(size):
    shape = np.array([10, 100, 1000, 10000, 100000])
    x = np.random.randint(0, 2, shape[size])
    y = np.random.randint(0, 2, shape[size])
    return (x, y)
