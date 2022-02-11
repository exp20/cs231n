from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

       
        "вычисление оценок классов"
        scores_l1 = X.dot(W1) + b1    # (N,D)x(D,H) = (N,H)
        scores_l1_relu = np.copy(scores_l1)
        scores_l1_relu[scores_l1_relu<0] = 0 # relu
        scores_l2 = scores_l1_relu.dot(W2) + b2    # (N,H)x(H,C) = (N,C)
        scores = scores_l2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        '''
        softmax loss вычисляется для каждой строки отдельно.
        чтобы увеличить численную стабильность вычисления softmax (могут быть очень большие значения exp)
        добавим нормировку на константу которая = максимальной оценке по классам в строке, для каждого сэмпла данных (строки)
        вычтем из каждой строки оценок соответствующую константу для этой строки.
        '''
        'нормировочные константы'
        max_line_scores = np.max(scores,axis = 1).reshape((-1,1)) # (1,N) -> (N,1) 
        scores -= max_line_scores 
        scores_exp = np.exp(scores)
        scores_yi = scores_exp[np.arange(0,len(y)),y] # (1,N)

        data_loss_local_arr = scores_yi/np.sum(scores_exp,axis = 1) # (1,N) / (1,N)
    
        data_loss = (-1/len(y))*np.sum(np.log(data_loss_local_arr + 1e-11))
        
        'regularization loss'
        reg_loss = reg*(np.sum(W1*W1)+np.sum(W2*W2))

        'data loss + reg loss'
        loss = data_loss + reg_loss

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        'градиент softmax loss dL/dD, где D - матрица на выходе 2ого слоя нейронов'
        probabilities = scores_exp/np.sum(scores_exp,axis = 1).reshape((-1,1)) # (N,C)
        probabilities_with_coef_y = probabilities
        probabilities_with_coef_y[range(0,len(y)),y] -=1 # (N,C)
        dD = probabilities_with_coef_y  # (N,C)
        dD  /= len(y)
        # db2 - градиент вектора смещений b2
        db2 = np.sum(dD,axis = 0 ) # (1,C)
 
        '''
        вычисление локальных градиентов для операции B*W2 = D:
        dW2, dB, где B = scores_l1_relu (выход после relu) (N,H).
        D (N,C); B (N,H); W2 (H,C)
        D = BW2 => 
          dD/dB = W2.T
          dD/dW2 = B.T
        восходящий градиент: dD (N,C)
        '''
        dW2 = np.dot(scores_l1_relu.T,dD) #  (H,N)x(N,C) = (H,C)
        dB = np.dot(dD,W2.T) # (N,C)x(C,H) = (N,H)

        '''
        вычисление локальных градиентов для операции B = max(0,A) (N,H):
        dA (N,H)
        восходящий градиент: dB
        высчитывается без расчета локального градиента и цепного правила
        '''
        dB[scores_l1<0]=0 # (N,H)
        dA = dB

        '''
        вычисление локальных градиентов для операции A = XW1:
         dW1, db1
        A (N,H); W1 (D,H); b1 (1,H); X (N,D)
        A = X*W1 => 
          dA/dW1 = X.T
        восходящий градиент: dA (N,H)
        '''
        dW1 = np.dot(X.T,dA) # (D,N)x(N,H) = (D,H)
        # db1 - градиент вектора смещений b1
        db1 = np.sum(dA,axis = 0) # (1,H)
        
        '''
        Вычислим dW1, dW2 от reg loss
        reg loss = reg*R(W1,W2) = reg(L2(W1)+L2(W2)), L2 - норма
        '''
        dW1 += reg*2*W1
        dW2 += reg*2*W2

        grads["W1"] = dW1
        grads["b1"] = db1
        grads["W2"] = dW2
        grads["b2"] = db2
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            sample_random_numbers = np.random.choice(range(0,len(X)), batch_size)
            X_batch = X[sample_random_numbers]
            y_batch = y[sample_random_numbers]
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            self.params["W1"] -= learning_rate*grads["W1"]
            self.params["W2"] -= learning_rate*grads["W2"]
            self.params["b1"] -= learning_rate*grads["b1"]
            self.params["b2"] -= learning_rate*grads["b2"]
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # прямой проход
        l1 = X.dot(self.params["W1"])+self.params["b1"] # (N,D)x(D,H) + для каждой сроки (1,H) = (N,H)
        l1[l1<0] = 0 # ReLU
        l2 = l1.dot(self.params["W2"]) + self.params["b2"] # (N,H)x(H,C) + для каждой строки (1,C) = (N,C)
        y_pred = np.argmax(l2,axis = 1) # (N,1)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
