from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W1 = weight_scale * np.random.randn(input_dim,hidden_dim) # D[a*X] = a^2 D[X] => std[a*X] = |a|std[X]
        b1 = np.zeros(hidden_dim)

        W2 = weight_scale * np.random.randn(hidden_dim,num_classes)
        b2 = np.zeros(num_classes)

        self.params = {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2}

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # архитектура: affien -> ReLU -> affine -> ReLU -> softmax

        out_1, cache_1 = affine_relu_forward(X, self.params["W1"], self.params["b1"])

        scores, cache_2 = affine_relu_forward(out_1, self.params["W2"], self.params["b2"])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        data_loss, dscores = softmax_loss(scores,y)
        loss = data_loss + self.reg*0.5*(np.sum(self.params["W1"]**2)+np.sum(self.params["W2"]**2)) # 0.5 - чтоб градиент был пропорц reg а не 2*reg

        dW1_reg = self.reg*self.params["W1"] # нет 2* так как добавили 0.5 в ошибку регуляризации
        dW2_reg = self.reg*self.params["W2"]

        dout1, dW2, db2 = affine_relu_backward(dscores, cache_2)
        dx, dW1, db1 = affine_relu_backward(dout1, cache_1)

        dW1 +=dW1_reg
        dW2 +=dW2_reg

        grads = {"W1":dW1, "b1": db1, "W2": dW2, "b2":db2}

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        prev_layer_size = input_dim
        i = 1
        for layer_size in hidden_dims: # только скрытые слои
          self.params["W{0}".format(i)] = weight_scale * np.random.randn(prev_layer_size, layer_size)
          self.params["b{0}".format(i)] = np.zeros(layer_size)
          prev_layer_size = layer_size
          i+=1
        # выходной слой перед softmax
        self.params["W{0}".format(i)] = weight_scale * np.random.randn(prev_layer_size, num_classes)
        self.params["b{0}".format(i)] = np.zeros(num_classes)

        if((self.normalization == "batchnorm") or (self.normalization == "layernorm")):
          i = 1
          for layer_size in hidden_dims: # на последнем слое нет нормализации
            self.params["gamma{0}".format(i)] = np.ones(layer_size)
            self.params["beta{0}".format(i)] = np.zeros(layer_size)
            i+=1

        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        affine_relu_caches = {} # кэш affine_relu_forward для каждого слоя
        batch_norm_caches = {} # кэш слоя батч нормализации, содержит значения необходимые для вычисления градиента
        layer_norm_caches = {} # кэш для нормализации по слою, содержит значения необходимые для вычисления градиента
        dropout_caches = {}
        layer_input = X

        if(self.normalization is None):
          for layer_number in range(0,self.num_layers-1):
            # affine + relu
            curr_layer_out, curr_layer_cache = affine_relu_forward(layer_input, self.params["W{0}".format(layer_number+1)],
                                                              self.params["b{0}".format(layer_number+1)]) # +1 т.к W1 W2 W3 ...
            affine_relu_caches[layer_number] = curr_layer_cache
            layer_input = curr_layer_out
            if(self.use_dropout):
              layer_input, dropout_cache = dropout_forward(curr_layer_out, self.dropout_param)
              dropout_caches[layer_number] = dropout_cache
              

        if(self.normalization == "batchnorm"):
          for layer_number in range(0,self.num_layers-1):
            batchnorm_out, batchnorm_cache = affine_batchnorm_relu_forward(layer_input,
                                              self.params["W{0}".format(layer_number+1)],
                                              self.params["b{0}".format(layer_number+1)],
                                              self.params["gamma{0}".format(layer_number+1)], 
                                              self.params["beta{0}".format(layer_number+1)], 
                                              self.bn_params[layer_number])
            batch_norm_caches[layer_number] = batchnorm_cache
            layer_input = batchnorm_out
            if(self.use_dropout):
              layer_input, dropout_cache = dropout_forward(batchnorm_out, self.dropout_param)
              dropout_caches[layer_number] = dropout_cache

        if(self.normalization == "layernorm"):
          for layer_number in range(0,self.num_layers-1):
            layernorm_out, layernorm_cache = affine_layernorm_relu_forward(layer_input,
                                              self.params["W{0}".format(layer_number+1)],
                                              self.params["b{0}".format(layer_number+1)],
                                              self.params["gamma{0}".format(layer_number+1)], 
                                              self.params["beta{0}".format(layer_number+1)], 
                                              self.bn_params[layer_number])
            layer_norm_caches[layer_number] = layernorm_cache
            layer_input = layernorm_out
            if(self.use_dropout):
              layer_input, dropout_cache = dropout_forward(layernorm_out, self.dropout_param)
              dropout_caches[layer_number] = dropout_cache

    
        # последний слой перед softmax без ReLU !!!
        scores, last_l_cache = affine_forward(layer_input, self.params["W{0}".format(self.num_layers)], 
                                                    self.params["b{0}".format(self.num_layers)])
        affine_relu_caches[self.num_layers-1] = last_l_cache                                               

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        data_loss, dscores = softmax_loss(scores,y)

        # подсчет reg_loss
        W_square_sum = 0
        for i in range(1,self.num_layers+1):
          W_square_sum += np.sum(self.params["W{0}".format(i)]**2)
          grads["W{0}".format(i)] = self.reg*self.params["W{0}".format(i)] # 2*0.5 = 1
        reg_loss = self.reg*0.5*W_square_sum

        loss = data_loss + reg_loss

        upstream_gradient = dscores
        dx, dw, db = affine_backward(upstream_gradient,last_l_cache) # последний слой перед softmax и он без ReLU
        grads["W{0}".format(self.num_layers)] += dw # так как там уже dreg_loss/dw
        grads["b{0}".format(self.num_layers)] = db
        upstream_gradient = dx

        if(self.normalization == "batchnorm"):
          for layer_number in range (self.num_layers-1,0,-1): #  последнее число будет 1 
            if(self.use_dropout):
              upstream_gradient = dropout_backward(upstream_gradient, dropout_caches[layer_number-1])
            dx, dw, db, dgamma, dbeta =  affine_batchnorm_relu_backward(upstream_gradient, batch_norm_caches[layer_number-1]) # -1 так как там индексация с 0
            upstream_gradient = dx
            grads["W{0}".format(layer_number)] += dw # так как там уже dreg_loss/dw
            grads["b{0}".format(layer_number)] = db
            grads["gamma{0}".format(layer_number)] = dgamma
            grads["beta{0}".format(layer_number)] = dbeta
            
            

        if(self.normalization == "layernorm"):
          for layer_number in range (self.num_layers-1,0,-1): #  последнее число будет 1 
            if(self.use_dropout):
              upstream_gradient = dropout_backward(upstream_gradient, dropout_caches[layer_number-1])
            dx, dw, db, dgamma, dbeta =  affine_layernorm_relu_backward(upstream_gradient, layer_norm_caches[layer_number-1]) # -1 так как там индексация с 0
            upstream_gradient = dx
            grads["W{0}".format(layer_number)] += dw # так как там уже dreg_loss/dw
            grads["b{0}".format(layer_number)] = db
            grads["gamma{0}".format(layer_number)] = dgamma
            grads["beta{0}".format(layer_number)] = dbeta

        if(self.normalization is None): 
          for layer_number in range (self.num_layers-1,0,-1): #  последнее число будет 1 
            if(self.use_dropout):
              upstream_gradient = dropout_backward(upstream_gradient, dropout_caches[layer_number-1])
            dx, dw, db = affine_relu_backward(upstream_gradient, affine_relu_caches[layer_number-1]) # -1 так как там индексация с 0
            upstream_gradient = dx
            grads["W{0}".format(layer_number)] += dw # так как там уже dreg_loss/dw
            grads["b{0}".format(layer_number)] = db
            

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
  # affine -> batchnorm -> relu
  a, fc_cache = affine_forward(x, w, b)
  bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(bn)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache

def affine_batchnorm_relu_backward(dout, cache):
  fc_cache, bn_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  db, dgamma, dbeta = batchnorm_backward_alt(da,bn_cache)
  dx, dw, db = affine_backward(db, fc_cache)
  return dx, dw, db, dgamma, dbeta

def affine_layernorm_relu_forward(x, w, b, gamma, beta, ln_param):
  # affine -> layernorm -> relu
  a, fc_cache = affine_forward(x, w, b)
  ln, ln_cache = layernorm_forward(a, gamma, beta, ln_param)
  out, relu_cache = relu_forward(ln)
  cache = (fc_cache, ln_cache, relu_cache)
  return out, cache

def affine_layernorm_relu_backward(dout, cache):
  fc_cache, ln_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dl, dgamma, dbeta = layernorm_backward(da,ln_cache)
  dx, dw, db = affine_backward(dl, fc_cache)
  return dx, dw, db, dgamma, dbeta
