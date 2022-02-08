from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    scores = X.dot(W) # (N,D)x(D,C)=(NxC)
    # Для обспечения стабильного вычисления экспоненты добавим константу определенную как -max(s1, s2, ... sn) максимум из оценок со знаком минус
    Cnst = np.max(scores,axis=1)
    scores -=Cnst[:,np.newaxis]
    for i in range(len(y)): # по объектам обучающей выборки
        softmax = np.exp(scores[i])/np.sum(np.exp(scores[i]))
        loss -= np.log(softmax[y[i]])# l_i
        #Gradients
        for j in range(num_classes):
            dW[:,j] += X[i] * softmax[j]
        dW[:,y[i]] -= X[i]
        
    loss /=len(y) # avg(L) : data loss
    loss += reg*np.sum(W*W) # loss + regularization loss
    
    dW /= len(y)
    dW += reg * 2 * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    '''
     W: A numpy array of shape (D, C) containing weights.
     X: A numpy array of shape (N, D) containing a minibatch of data.
     y: A numpy array of shape (N,)
    '''
    # dL_i/d_w_yi = -yi(j==y_i)+exp(s_yi)/sum(exp(s_j))
    scores = X.dot(W) # (N,D)x(D,C) = (N,C)
    # обеспечение числовой стабильности вычисления экспонент
    scores -= np.max(scores, axis=1)[:, np.newaxis] # вычитание из матрицы вектора столбца (благодаря добавлению размерности np.newaxis)
    probabilities = np.exp(scores)/np.sum(np.exp(scores), axis = 1)[:,np.newaxis] # деление матрицы на вектор столбец (благодаря добавления размерности np.newaxis)
    loss = (1/len(y))*np.sum(-1*np.log(probabilities[np.arange(len(y)),y]))+reg*np.sum(W*W)
    
    # Вычисление градиента
    soft_max_matrix = probabilities
    soft_max_matrix[np.arange(len(y)),y] -=1 # коэффы при p_yi (вероятностях целегого класса),
    soft_max_matrix /=len(y)
    
    # сумма производных в Li
    dW = X.T.dot(soft_max_matrix) # (D,N)x(N,C)=(D,C)
    dW+= 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
