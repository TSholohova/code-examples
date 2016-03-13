import numpy as np


# v1_vector(X : 2d np.array of int or double,
#           at least one a non-zero element on the diagonal)
# returns int or double
def v1_vector(X):
    y = np.diag(X)
    return np.prod(y[np.nonzero(y)])


# v2_non_vector(X : 2d np.array of int or double,
#               at least one a non-zero element on the diagonal)
# returns int or double
def v2_non_vector(X):
    res = 1
    for i in range(min(X.shape[0], X.shape[1])):
        if X[i, i] != 0:
            res *= X[i, i]
    return res


# v3_part_vector(X : 2d np.array of int or double,
#                at least one a non-zero element on the diagonal)
# returns int or double
def v3_part_vector(X):
    res = 1
    y = np.diag(X)
    for i in y:
        if i != 0:
            res *= i
    return res


# gen(0 <= size <= 4: int) return tuple(X : 2d np.array of int, )
def gen(size):
    shapes = np.array([(50, 30), (100, 120),
                       (520, 500), (1000, 1020),
                       (5020, 5000)])
    X = np.random.randint(-1, 1, shapes[size])
    return (X, )
