import scipy.spatial
import numpy as np


# v1_vector(X, Y : 2d np.arrays of double, X.shape[1] == Y.shape[1])
# returns (distances : 1d np.array of double)
def v1_vector(X, Y):
    return np.array([np.sqrt(np.sum((Y - X[i])**2, axis=1))
                    for i in range(0, X.shape[0])])


# v2_non_vector(X, Y : 2d np.arrays of double, X.shape[1] == Y.shape[1])
# returns (distances : 1d np.array of double)
def v2_non_vector(X, Y):
    distances = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(0, X.shape[0]):
        for j in range(0, Y.shape[0]):
            distances[i, j] = np.sqrt(np.sum((X[i] - Y[j])**2))
    return distances


# v3_part_vector(X, Y : 2d np.arrays of double, X.shape[1] == Y.shape[1])
# returns (distances : 1d np.array of double)
def v3_part_vector(X, Y):
    return scipy.spatial.distance.cdist(X, Y)


# gen(0 <= size <= 4 : int) returns (X, Y) : np.arrays of double
def gen(size):
    shape_X = np.array([1, 10, 100, 1000, 10000])
    shape_Y = np.array([10, 10, 10, 10, 10])
    dim = np.array([13, 8, 5, 3, 2])
    X = np.random.uniform(-10, 10, (shape_X[size], dim[size]))
    Y = np.random.uniform(-10, 10, (shape_Y[size], dim[size]))
    return (X, Y)
