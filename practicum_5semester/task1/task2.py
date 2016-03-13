import numpy as np


# v1_vector(X : 2d np.array of int or double,
#           i, j : 1d np.arrays of int, i.shape[0] == j.shape[0]
#           0 <= values of i < X.shape[0], 0 <= values of j < X.shape[1]
# returns (1d np.array of int or double))
def v1_vector(X, i, j):
    return X[i, j]


# v2_non_vector(X : 2d np.array of int or double,
#           i, j : 1d np.arrays of int, i.shape[0] == j.shape[0]
#           0 <= values of i < X.shape[0], 0 <= values of j < X.shape[1]
# returns (list of int or double))
def v2_non_vector(X, i, j):
    res = []
    for k in range(0, i.shape[0]):
        res.append(X[i[k], j[k]])
    return res


# v3_part_vector(X : 2d np.array of int or double,
#           i, j : 1d np.arrays of int, i.shape[0] == j.shape[0]
#           0 <= values of i < X.shape[0], 0 <= values of j < X.shape[1]
# returns (1d np.array of int or double))
def v3_part_vector(X, i, j):
    x = X[i]
    return x[np.arange(0, j.shape[0]), j]


# gen(0 <= size <= 4: int) returns correct (X, i, j) : np.arrays of int
def gen(size):
    shapes_X = np.array([(70, 50), (100, 120),
                         (520, 500), (1000, 1020),
                         (5020, 5000)])
    shapes_ij = np.array([60, 110, 510, 1010, 5010])
    i = np.random.randint(0, shapes_X[size][0] - 1, shapes_ij[size])
    j = np.random.randint(0, shapes_X[size][1] - 1, shapes_ij[size])
    X = np.random.randint(-100, 100, shapes_X[size])
    return (X, i, j)
