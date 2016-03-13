import scipy.stats
import numpy as np


# v1_vector(X: 2d np.array of double, X.shape=(N, D),
#                m: 1d np.array of double, m.shape=(D),
#                C: 2d np.array of double, C.shape=(D, D))
# returns (distances : 1d np.array of double)
def v1_vector(X, m, C):
    n = m.shape[0]
    ans = -(n/2.0)*np.log(2*np.pi) - 0.5*np.linalg.slogdet(C)[1]
    ans -= 0.5*np.dot(np.dot((X-m), np.linalg.inv(C)), (X-m).T)
    return np.diag(ans)


# v2_non_vector(X: 2d np.array of double, X.shape=(N, D),
#                m: 1d np.array of double, m.shape=(D),
#                C: 2d np.array of double, C.shape=(D, D))
# returns (distances : 1d np.array of double)
def v2_non_vector(X, m, C):
    ans = []
    n = m.shape[0]
    for i in range(X.shape[0]):
        x = X[i]
        ans.append(-(n/2.0)*np.log(2*np.pi) - 0.5*np.linalg.slogdet(C)[1] -
                   0.5*np.dot(np.dot((x-m), np.linalg.inv(C)), (x-m).T))
    return np.array(ans)


# v3_part_vector(X: 2d np.array of double, X.shape=(N, D),
#                m: 1d np.array of double, m.shape=(D),
#                C: 2d np.array of double, C.shape=(D, D))
# returns (distances : 1d np.array of double)
def v3_part_vector(X, m, C):
    return scipy.stats.multivariate_normal(m, C).logpdf(X)


# gen(0 <= size <= 4 : int) returns (X, m, C)
def gen(size):
    shape = np.array([5, 10, 20, 40, 80])
    X = np.random.uniform(-2.0, 2.0, (shape[size], shape[size]))
    m = np.random.uniform(-1.0, 1.0, shape[size])
    C = np.eye(shape[size])
    return (X, m, C)
