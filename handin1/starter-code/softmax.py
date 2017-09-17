import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import scipy.special as scisp
import scipy.optimize as opti
import cProfile, pstats, io
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from h1_util import numerical_grad_check

def softmax(X):
    """ 
    Compute the softmax of each row of an input matrix (2D numpy array).
    
    the numpy functions amax, log, exp, sum may come in handy as well as the keepdims=True option and the axis option.
    Remember to handle the numerical problems as discussed in the description.
    You should compute lg softmax first and then exponentiate 
    
    More precisely this is what you must do.
    
    For each row x do:
    compute max of x
    compute the log of the denominator for softmax but subtracting out the max i.e (log sum exp x-max) + max
    compute log of the softmax: x - lsm
    exponentiate that
    
    You can do all of it without for loops using numpys vectorized operations.

    Args:
        X: numpy array shape (n, d) each row is a data point
    Returns:
        res: numpy array shape (n, d)  where each row is the softmax transformation of the corresponding row in X i.e res[i, :] = softmax(X[i, :])
    """
    res = np.zeros(X.shape)
    ### YOUR CODE HERE no for loops please
    X_max = np.amax(X)
    denom = np.log(np.sum(np.exp(X - X_max), keepdims=True, axis=1)) + X_max
    res = np.exp(X - denom)
    ### END CODE
    return res
    
def soft_cost(X, Y, W, reg=0.0):
    """ 
    Compute the regularized cross entropy and the gradient under the softmax model 
    using data X,y and weight vector w.
    Remember not to regualarize the bias parameters which are the first index of each model i.e. the first row of W.

    the functions np.nonzero, np.sum, np.dot (@), may come in handy
    Args:
        X: numpy array shape (n, d) - the data each row is a data point
        Y: numpy array shape (n, k) target values in 1-in-K encoding (n x K)
        W: numpy array shape (d x K)
        reg: scalar - Optional regularization parameter
    Returns:
        totalcost: Average Negative Log Likelihood of w + reg cost 
        gradient: The gradient of the average Negative Log Likelihood at w + gradient of regularization
    """
    input_size = X.shape[0]
    # W = W.reshape(X.shape[1],-1)
    cost = np.nan
    grad = np.zeros(W.shape)*np.nan
    ### YOUR CODE HERE
    for i in range(0, input_size):
        x = X[i]
        y = Y[i]
        j = np.nonzero(y)
        w_dot_x = np.dot(np.transpose(W), x)
        cost += -(w_dot_x[j] - np.sum(np.exp(w_dot_x)))
        #cost += np.dot(y, np.log(softmax(np.dot(np.transpose(W), x))))
    cost = (-cost)/input_size
    l2reg = 0.5 * reg * np.sum(np.dot(np.transpose(W[1:]), W[1:]))
    cost += l2reg
    grad = (-np.transpose(X) @ (Y - softmax(X @ W)))/input_size
    ### END CODE
    return cost, grad

def batch_grad_descent(X, Y, W=None, reg=0.0, lr=0.5, rounds=10):
    """
    Run batch gradient descent to learn softmax regression weights that minimize the in-sample error
    Args:
        X: numpy array shape (n, d) - the data each row is a data point
        Y: numpy array shape (n, k) target values in 1-in-K encoding (n x K)
        W: numpy array shape (d x K)
        reg: scalar - Optional regularization parameter
        lr: scalar - learning rate

    Returns: 
        w: numpy array shape (d,) learned weight vector w
    """
    if W is None: W = np.zeros((X.shape[1], Y.shape[1]))
    ### YOUR CODE HERE
    ### END CODE
    return W

def mini_batch_grad_descent(X, Y, W=None, reg=0.0, lr=0.1, epochs=10, batch_size=16):
    """
    Run Mini-Batch Gradient Descent on data X,Y to minimize the NLL for softmax regression.
    Printing the performance every epoch is a good idea
    
    Args:
        X: numpy array shape (n, d) - the data each row is a data point
        Y: numpy array shape (n, k) target values in 1-in-K encoding (n x K)
        W: numpy array shape (d x K)
        reg: scalar - Optional regularization parameter
        lr: scalar - learning rate
        batchsize: scalar - size of mini-batch
        epochs: scalar - number of iterations through the data to use

    Returns: 
        w: numpy array shape (d,) learned weight vector w
    """
    if W is None: W = np.zeros((X.shape[1],Y.shape[1]))
    ### YOUR CODE HERE
    ### END CODE
    return W

def test_softmax():
    print('Test softmax')
    X = np.zeros((3,2))
    X[0, 0] = np.log(4)
    X[1, 1] = np.log(2)
    print('Input to Softmax: \n', X)
    sm = softmax(X)
    expected = np.array([[4.0/5.0, 1.0/5.0], [1.0/3.0, 2.0/3.0], [0.5, 0.5]])
    print('Result of softmax: \n', sm)
    assert np.allclose(expected, sm), 'Expected {0} - got {1}'.format(expected, sm)
    print('Test complete')
        

def test_reg_grad():
    print('*'*5, 'Testing  Gradient')
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]])    
    w = np.ones((2, 3))
    y = np.eye(3, dtype='int64')
    reg = 1.0
    f = lambda z: soft_cost(X, y, W=z, reg=reg)
    numerical_grad_check(f, w)
    print('Test Success')

    
if __name__ == "__main__":
    test_softmax()
    test_reg_grad()
