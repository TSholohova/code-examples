import numpy as np
import math


def initialize(hidden_size, visible_size):    
    res = np.array([])
    n_in = visible_size
    for n_out in np.concatenate((hidden_size, np.array([visible_size]))):
        low = -math.sqrt(6 / (n_in + n_out + 1))
        high = math.sqrt(6 / (n_in + n_out + 1))        
        w = np.random.uniform(low, high, (n_in, n_out))
        res = np.concatenate((res, w.reshape(n_in * n_out), np.zeros(n_out)))
        n_in = n_out
    return res

def autoencoder_loss(theta, visible_size, hidden_size, lambda_, sparsity_param, beta, data):
    W = theta
    n_in = visible_size
    i = 0
    m = 0
    X_linear = []
    X_sigm = []
    X_ro = []    
    x = data    
    reg1 = 0.0
    reg2 = 0.0
    for n_out in np.concatenate((hidden_size, np.array([visible_size]))):
        w = W[i:i+n_in*n_out].reshape(n_in, n_out)
        reg1 += np.sum(w**2, axis=(0, 1))        
        b = W[i+n_in*n_out:i+(n_in+1)*n_out]
        i += (n_in+1)*n_out
        x = np.dot(x, w) + b
        X_linear.append(x)
        x = 1.0 / (1.0 + np.exp(-x))
        X_sigm.append(x)
        ro = np.mean(x, axis=0)
        X_ro.append(ro)        
        if m+1 == (hidden_size.shape[0] + 2)//2:
            reg2 += np.sum(sparsity_param*np.log(sparsity_param/ro) + \
                          (1-sparsity_param)*np.log((1-sparsity_param)/(1-ro)))
        n_in = n_out
        m += 1
        
    L = 1.0 / (2.0 * data.shape[0]) * np.sum((data-x)**2, axis=(0, 1))
    L += (lambda_ / 2.0) * reg1
    L += beta * reg2
    grad = np.zeros(theta.shape)    
    dLdy = (x - data) / data.shape[0]
    m = len(X_sigm)-1
    n_out = visible_size
    i = W.size    
    for n_in in np.concatenate((hidden_size, np.array([visible_size]))):
        ro = X_ro[m]        
        if m+1 == (hidden_size.shape[0] + 2)//2:
            dLdy += 1/data.shape[0] * beta * ((1-sparsity_param)/(1-ro) - sparsity_param/ro)        
        dLdy = -X_sigm[m] * (X_sigm[m] - 1) * dLdy
        if m >= 1:
            x = X_sigm[m-1]
        else:
            x = data
        i -= (n_in + 1) * n_out
        w = W[i:i+n_in*n_out].reshape(n_in, n_out)
        b = W[i+n_in*n_out:i+(n_in+1)*n_out]
        dLdw = np.dot(x.T, dLdy) + lambda_*w
        dLdb = np.sum(dLdy, axis=0)
        grad[i:i+n_in*n_out] = dLdw.ravel()
        grad[i+n_in*n_out:i+(n_in+1)*n_out] = dLdb
        dLdy = np.dot(dLdy, w.T)
        n_out = n_in
        m -= 1
        
    return (L, grad)
    
def autoencoder_transform(theta, visible_size, hidden_size, layer_number, data):
    layer = np.concatenate((hidden_size, np.array([visible_size])))[:layer_number]
    n_in = visible_size
    W = theta
    i = 0
    x = data    
    for n_out in layer:
        w = W[i:i+n_in*n_out].reshape(n_in, n_out)
        b = W[i+n_in*n_out:i+(n_in+1)*n_out]
        i += (n_in+1)*n_out
        x = np.dot(x, w) + b
        x = 1 / (1 + np.exp(-x))
        n_in = n_out
    return x