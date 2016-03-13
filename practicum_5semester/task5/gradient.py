import math
import numpy as np


def compute_gradient(J, theta):
    d = theta.shape[0]    
    eps = 1e-6
    res = np.zeros(d)
    for x in range(d):
        mask = np.zeros(d)
        mask[x] = 1        
        res[x] = (J(theta + eps * mask) - J(theta - eps * mask)) / (2 * eps)
    return res


def J1(x):
    return math.sin(x[0] * x[1] * x[2]) + x[0] * x[2]


def dJ1(x):
    return np.array([x[1]*x[2]*math.cos(x[0]*x[1]*x[2]) + x[2],
                     x[2]*x[0]*math.cos(x[0]*x[1]*x[2]),
                     x[0]*x[1]*math.cos(x[0]*x[1]*x[2]) + x[0]])

def check(J, dJ, theta0, theta1, theta_step):
    eps = 1e-4
    res = True
    x = theta0
    while np.sum(x < theta1) != 0:
        if np.linalg.norm(compute_gradient(J, x) - dJ(x)) > eps:
            print(x, compute_gradient(J, x), dJ(x))
            res = False
        x += theta_step
    return res


def check_gradient():
    if check(J1, dJ1, np.array([-5.0, -6.0, -7.0]), np.array([5.0, 4.0, 4.0]), np.array([0.3, 0.2, 0.4])):
        print('Ok')
    else:
        print('Fail')