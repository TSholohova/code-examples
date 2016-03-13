import numpy as np
import cvxopt
import time
import matplotlib.pyplot as plt
import sklearn.svm


def svm_qp_primal_solver(X, y, C, tol=1e-6, max_iter=100, verbose=False):
    cvxopt.solvers.options['abstol'] = tol
    cvxopt.solvers.options['maxiters'] = max_iter
    cvxopt.solvers.options['show_progress'] = verbose

    d = X.shape[1]
    L = X.shape[0]

    P = np.eye(1+d+L, 1+d+L)
    P[0, 0] = 0
    P[range(1+d, 1+d+L), range(1+d, 1+d+L)] = 0

    q = np.zeros(1+d+L)
    q[1+d:] = C

    G = np.zeros((L+L, 1+d+L))
    G[:L, 0] = -y
    G[:L, 1:1+d] = -y[:, np.newaxis] * X
    G[:L, 1+d:] = -np.eye(L, L)
    G[L:, 1+d:] = -np.eye(L, L)

    h = np.zeros(2*L)
    h[:L] = -1

    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)

    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)

    now = time.clock()
    res = cvxopt.solvers.qp(P, q, G, h)
    res_time = time.clock() - now

    return {'w0': res['x'][0],
            'w': np.array(res['x'][1:d+1]).ravel(),
            'status': int(res['status'] != 'optimal'),
            'time': res_time}


def svm_qp_dual_solver(X, y, C, tol=1e-6, max_iter=100,
                       verbose=False, gamma=0):
    cvxopt.solvers.options['abstol'] = tol
    cvxopt.solvers.options['maxiters'] = max_iter
    cvxopt.solvers.options['show_progress'] = verbose

    L = X.shape[0]

    if gamma == 0:
        K = np.dot(X, X.T)
    else:
        K = np.sum(X**2, axis=1)[:, np.newaxis] - 2.0*np.dot(X, X.T) + \
            np.sum(X**2, axis=1)[np.newaxis, :]
        K = np.exp(-gamma*K)

    P = y[np.newaxis, :] * y[:, np.newaxis] * K
    q = -np.ones(L)

    G = np.vstack((np.eye(L, L), -np.eye(L, L)))
    h = np.concatenate((C * np.ones(L), np.zeros(L)))[:, np.newaxis]

    A = 1.0 * y[np.newaxis, :]
    b = np.array([[0.0]])

    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    A = cvxopt.matrix(A)
    b = cvxopt.matrix(b)

    now = time.clock()
    res = cvxopt.solvers.qp(P, q, G, h, A, b)
    res_time = time.clock() - now

    return {'w0': res['y'][0],
            'w': compute_w(X, y, np.array(res['x']).ravel()),
            'A': np.array(res['x']).ravel(),
            'status': int(res['status'] != 'optimal'),
            'time': res_time}


def svm_liblinear_solver(X, y, C, tol=1e-6, max_iter=100, verbose=False):
    svm = sklearn.svm.LinearSVC(loss='hinge', tol=tol, C=C, verbose=verbose,
                                intercept_scaling=10, max_iter=max_iter)
    now = time.clock()
    svm.fit(X, y)
    res_time = time.clock() - now
    return {'w0': svm.intercept_[0],
            'w': svm.coef_.copy()[0],
            'time': res_time}


def svm_libsvm_solver(X, y, C, tol=1e-6, max_iter=100, verbose=False, gamma=0):
    if gamma == 0:
        svm = sklearn.svm.SVC(C=C, kernel='linear', tol=tol, verbose=verbose,
                              max_iter=max_iter)
    else:
        svm = sklearn.svm.SVC(C=C, kernel='rbf', gamma=gamma, tol=tol,
                              verbose=verbose, max_iter=max_iter)

    now = time.clock()
    svm.fit(X, y)
    res_time = time.clock() - now

    A = np.zeros(X.shape[0])
    A[svm.support_] = np.abs(svm.dual_coef_)

    return {'w0': svm.intercept_[0],
            'w': compute_w(X, y, A),
            'A': A,
            'time': res_time}


def compute_primal_objective(X, y, w0, w, C):
    return 1.0/2.0 * np.sum(w**2) + C * \
           np.sum(np.maximum(0, (1.0 - y*(np.dot(w[np.newaxis, :], X.T)+w0))))


def compute_dual_objective(X, y, A, C, gamma=0):
    if gamma == 0:
        K = np.dot(X, X.T)
    else:
        K = np.sum(X**2, axis=1)[:, np.newaxis] - 2.0*np.dot(X, X.T) + \
            np.sum(X**2, axis=1)[np.newaxis, :]
        K = np.exp(-gamma*K)
    K *= y[:, np.newaxis]*A[:, np.newaxis]*A[np.newaxis, :]*y[np.newaxis, :]
    return np.sum(A) - np.sum(K)/2.0


def svm_subgradient_solver(X, y, C, tol=1e-6, max_iter=100, verbose=False,
                           alpha=0.1, beta=0.7, size=0):
    L = X.shape[0]
    d = X.shape[1]

    if size == 0:
        size = L

    mean = np.mean(X, axis=0)
    X = X.copy() - mean[np.newaxis, :]

    status = 1
    w0 = 0.0
    w = np.ones(d) * 1.0
    now = time.clock()
    obj_curve = [compute_primal_objective(X, y, w0, w, C)]

    for i in range(max_iter):
        index = np.arange(0, L)
        np.random.shuffle(index)
        index = index[0:size]
        i_size = index.size
        Z = X[index, :]
        zy = y[index]
        flg = ((1.0 - zy*(np.dot(w[np.newaxis, :], Z.T) + w0)) > 1e-5)
        flg = flg.ravel()[:, np.newaxis]
        df_dw = i_size*(1.0/L)*w - C*np.sum(flg*zy[:, np.newaxis]*Z, axis=0)
        df_dw0 = -C*np.sum(flg*zy)

        # norm = np.sum(np.append(df_dw, df_dw0)**2.0)**(0.5)
        # df_dw /= norm
        # df_dw0 /= norm

        alpha_i = alpha / ((i+1)**beta)

        w -= alpha_i * df_dw
        w0 -= alpha_i * df_dw0

        # print(w, w0)

        obj_curve.append(compute_primal_objective(X, y, w0, w, C))

        if abs(obj_curve[-1]-obj_curve[-2]) < tol:
            status = 0
            break

    res_time = time.clock() - now
    w0 -= np.dot(w, mean)
    return {'w0': w0,
            'w': w,
            'status': status,
            'objective_curve': np.array(obj_curve),
            'time': res_time}


def compute_w(X, y, A):
    return np.sum(A[:, np.newaxis] * y[:, np.newaxis] * X, axis=0)


def compute_support_vectors(X, y, A):
    return X[A > 0+1e-6, :]


def visualize(X, y, w0=None, w=None, A=None, gamma=0):
    X1 = X[np.where(y < 0)[0], :]
    X2 = X[np.where(y > 0)[0], :]
    plt.scatter(X1[:, 0], X1[:, 1], c='b', s=25)
    plt.scatter(X2[:, 0], X2[:, 1], c='r', s=25)

    if w0 is None:
        plt.show()
        return

    size = 100

    x_min = np.min(X[:, 0]) - 1
    x_max = np.max(X[:, 0]) + 1
    y_min = np.min(X[:, 1]) - 1
    y_max = np.max(X[:, 1]) + 1

    xp = np.linspace(x_min, x_max, size)
    yp = np.linspace(y_min, y_max, size)
    xp, yp = np.meshgrid(xp, yp)

    if A is None:
        z = w[0]*xp + w[1]*yp + w0
    elif gamma == 0:
        z = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                v = np.array([xp[i, j], yp[i, j]])
                z[i, j] = np.sum(np.dot(y[:, np.newaxis] * A[:, np.newaxis] *
                                 X, v)) + w0
    else:
        z = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                v = np.array([xp[i, j], yp[i, j]])
                K = np.exp(-gamma*np.sum((X-v)**2, axis=1)) * A * y
                z[i, j] = np.sum(K) + w0

    if A is not None:
        sup_vect = compute_support_vectors(X, y, A)
        plt.scatter(sup_vect[:, 0], sup_vect[:, 1], c='k', marker='x', s=70,
                    linewidth=1)

    plt.contour(xp, yp, z, levels=[0.0, -1.0, 1.0], colors=['k', 'b', 'r'],
                linestyles=['-', '--', '--'])
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.show()
    return
