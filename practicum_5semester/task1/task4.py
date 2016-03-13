import numpy as np


# v1_vector(x : 1d np.array of int or double)
# returns (int or double)
def v1_vector(x):
    return np.max(x[np.where(x[0:-1] == 0)[0] + 1])


# v2_non_vector(x : 1d np.array of int or double)
# returns (int or double)
def v2_non_vector(x):
    res = -np.inf
    for i in range(1, len(x)):
        if (x[i - 1] == 0) and (x[i] > res):
            res = x[i]
    return res


# v3_part_vector(x : 1d np.array of int or double)
# returns (int or double)
def v3_part_vector(x):
    x = x[np.where(x[0:-1] == 0)[0] + 1]
    res = -np.inf
    for i in x:
        if res < i:
            res = i
    return res


# gen(0 <= size <= 4: int) return tuple(x : 1d np.array of int, )
def gen(size):
    shape = np.array([10, 100, 1000, 10000, 100000])
    x = np.random.randint(0, 5, shape[size])
    mask = np.random.randint(0, shape[size] - 1, shape[size] / 5)
    x[mask] = 0
    return (x,)
