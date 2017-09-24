import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from sklearn.metrics import confusion_matrix
from h1_util import numerical_grad_check

np.seterr(all='raise')

def logistic(z):
    """ 
    Computes the logistic function 1/(1+e^{-x}) to each entry in input vector z.
    
    np.exp may come in handy
    Args:
        z: numpy array shape (d,) 
    Returns:
       logi: numpy array shape (d,) each entry transformed by the logistic function 
    """
    logi = np.zeros(z.shape)
    ### YOUR CODE HERE
    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    logi = np.vectorize(sigmoid)(z)
    ### END CODE
    assert logi.shape == z.shape
    return logi

def log_cost(X, y, w, reg=0):
    """
    Compute the (regularized) cross entropy and the gradient under the logistic regression model 
    using data X, targets y, weight vector w (and regularization reg)
    
    The L2 regularization is 1/2 reg*|w_{1,d}|^2 i.e. w[0], the bias, is not regularized
    
    np.log, np.sum, np.choose, np.dot may be useful here
    Args:
        X: np.array shape (n,d) float - Features 
        y: np.array shape (n,)  int - Labels 
        w: np.array shape (d,)  float - Initial parameter vector
        reg: scalar - regularization parameter

    Returns:
      cost: scalar the cross entropy cost of logistic regression with data X,y using regularization parameter reg
      grad: np.arrray shape(n,d) gradient of cost at w with regularization value reg
    """
    cost = 0
    grad = np.zeros(w.shape)
    ### YOUR CODE HERE
    n = X.shape[0]
    w_reg = np.insert(w[1:], 0, 0) # do not regularize bias
    
    # calculate cost
    score = logistic(np.dot(X, w))
    nll = - np.sum(y * np.log(score) + (1-y) * np.log(1 - score)); cost = nll/n
    
    # calculate gradient
    nll_grad = - np.dot(np.transpose(X), y - score); grad = nll_grad/n
    
    # regularize
    cost += 0.5 * reg * np.linalg.norm(w_reg, ord=2)
    grad += reg * w_reg
    ### END CODE
    assert grad.shape == w.shape
    return cost, grad

def batch_grad_descent(X, y, w=None, reg=0, lr=1.0, rounds=10):
    """
    Run Gradient Descent for logistic regression using all the data to compute gradient in each step
            
    Args:
        X: np.array shape (n,d) dtype float32 - Features 
        y: np.array shape (n,) dtype int32 - Labels 
        w: np.array shape (d,) dtype float32 - Initial parameter vector
        lr: scalar - learning rate for gradient descent
        reg: scalar regularization parameter (optional)    
        rounds: rounds to run (optional)

    Returns: 
        w: numpy array shape (d,) learned weight vector w
    """
    if w is None: w = np.zeros(X.shape[1])    
    #lr = 1.0
    ### YOUR CODE HERE
    for i in range(rounds):
        cost, grad = log_cost(X, y, w, reg)
        w = w - lr * grad # update hypothesis
    ### END CODE
    return w
    
def mini_batch_grad_descent(X, y, w=None, reg=0, lr=0.1, batch_size=16, epochs=10):
    """
      Run mini-batch stochastic Gradient Descent for logistic regression 
      use batch_size data points to compute gradient in each step.
    
    The function np.random.permutation may prove useful for shuffling the data before each epoch
    It is wise to print the performance of your algorithm at least after every epoch to see if progress is being made.
    Remeber the stochastic nature of the algorithm may give fluctuations in the cost as iterations increase.

    Args:
        X: np.array shape (n,d) dtype float32 - Features 
        y: np.array shape (n,) dtype int32 - Labels 
        w: np.array shape (d,) dtype float32 - Initial parameter vector
        lr: scalar - learning rate for gradient descent
        reg: scalar regularization parameter (optional)    
        rounds: rounds to run (optional)
        batch_size: number of elements to use in minibatch
        epochs: Number of scans through the data

    Returns: 
        w: numpy array shape (d,) learned weight vector w
    """
    if w is None: w = np.zeros(X.shape[1])
    ### YOUR CODE
    n = X.shape[0]
    batches, remainder = n // batch_size, n % batch_size
    for i in range(epochs):
        perm = np.random.permutation(n) # shuffle indices
        X_rand, y_rand = X[perm], y[perm] # shuffle X and y in unison
        for j in range(batches):
            rest = remainder if (j + 1 >= batches) else 0
            start, end = j*batch_size, j*batch_size + batch_size + rest
            X_batch, y_batch = X_rand[start:end], y_rand[start:end]
            cost, grad = log_cost(X_batch, y_batch, w, reg)
            w = w - lr * grad # update hypothesis
    ### END CODE
    return w

def fast_descent(X, y, w=None, reg=0, rounds=100):
    """ Uses fancy optimizer to do the gradient descent """
    # unstable if linear separable...
    if w is None: w = np.zeros(X.shape[1])
    w =  scipy.optimize.minimize(lambda t: log_cost(X,y,t,reg), w, jac=True, options={'maxiter':rounds, 'disp':True})
    return w.x


def test_logistic():
    print('*'*5, 'Testing logistic function')
    a = np.array([0,1,2,3])
    lg = logistic(a)
    target = np.array([ 0.5, 0.73105858, 0.88079708, 0.95257413])
    assert np.allclose(lg, target), 'Logistic Mismatch Expected {0} - Got {1}'.format(target, lg)
    a = np.array([-1, -2, -3, -4])
    lg = logistic(a)
    target = np.array([0.26894142, 0.11920292, 0.04742587, 0.01798620])
    assert np.allclose(lg, target), 'Logistic Mismatch Expected {0} - Got {1}'.format(target, lg)
    print('Test Success!')

def test_cost():
    print('*'*5, 'Testing Cost Function')
    X = np.array([[1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 0], dtype='int64')
    w = np.array([0.0, 0.0])
    reg = 0
    cost,_ = log_cost(X,y, w, reg)
    target = -np.log(0.5)
    assert np.allclose(cost, target), 'Cost Function Error:  Expected {0} - Got {1}'.format(target, cost)
    X = np.array([[2.0, 3.0, 5.0, 7.0], [9.0, 4.0, 0.0, 11.0]])
    y = np.array([0, 1], dtype='int64')
    w = np.array([4.0, 5.0, 3.0, 8.0])
    reg = 0
    cost,_ = log_cost(X,y, w, reg)
    target = 18.3684002848
    assert np.allclose(cost, target), 'Cost Function Error:  Expected {0} - Got {1}'.format(target, cost)
    print('Test Success')

def test_reg_cost():
    print('*'*5, 'Testing Regularized Cost Function')
    X = np.array([[1.0, -1.0], [-1.0, 1.0]])
    y = np.array([0, 0], dtype='int64')
    w = np.array([1.0, 1.0])
    reg = 1.0
    cost,_ = log_cost(X, y, w, reg=reg)
    target = -np.log(0.5) + 0.5
    assert np.allclose(cost, target), 'Cost Function Error:  Expected {0} - Got {1}'.format(target, cost)
    print('Test Success')

def test_grad():
    print('*'*5, 'Testing  Gradient')
    X = np.array([[1.0, 0.0], [1.0, 1.0]])
    w = np.array([0.0, 0.0])
    y = np.array([0, 0]).astype('int64')
    reg = 0
    f = lambda z: log_cost(X, y, w=z, reg=reg)
    numerical_grad_check(f, w)
    print('Test Success')

def test_reg_grad():
    print('*'*5, 'Testing  Gradient')
    X = np.array([[1.0, 0.0], [1.0, 1.0]])
    w = np.array([0.0, 0.0])
    y = np.array([0, 0]).astype('int64')
    reg = 1.0
    f = lambda z: log_cost(X, y, w=z, reg=reg)
    numerical_grad_check(f, w)
    print('Test Success')
    
def test_minibatch():
    print('*'*5, 'Testing Minibatch Gradient')
    X = np.array([[1.0, 0.0], [1.0, 1.0]])
    w = np.array([0.0, 1.0])
    y = np.array([0, 0]).astype('int64')
    reg = 0
    w_trained = mini_batch_grad_descent(X, y, w, reg, 0.1, batch_size=1, epochs=10)
    print(w_trained)
    print('Test Success')
    
if __name__ == '__main__':
    test_logistic()
    test_cost()
    test_grad()
    test_reg_cost()
    test_reg_grad()
    
    
