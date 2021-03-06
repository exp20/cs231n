from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # на входе пакет изображений, преобразования массива пикселей изображений в строку
    X = x.reshape((len(x),-1)) # (N,D)
    out = X.dot(w)+b # (N,D)x(D,M) = (N,M)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    X = x.reshape((len(x),-1)) # (N,D)

    dx = dout.dot(w.T) # (N,M)x(M,D)  = (N,D)
    dx = dx.reshape(x.shape) # возвращаем к первонгачальнйо размерности входа x

    dw = X.T.dot(dout) #  (D,N)x(N,M) = (D,M)
    db = np.sum(dout,axis = 0) 

   
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0,x) # возвращает новый массив содержащий maximum(0,a) по элементам

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dout[x<0]=0
    dx = dout

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        '''
        1) вычисление статистики для текущего батча
        2) нормализация признаков в батче по вычисленным статистикам c параметрами масштаба и сдвига
        3) обновление скользящих средних и дисперсий - эти значения будут использованы во время инференса (на тестовых данных)
        '''
        mx = np.mean(x,axis = 0) # (1,D)
        x_centered = x - mx # (N,D)
        x_centered_sq = x_centered**2
        Dx = np.mean(x_centered_sq,axis = 0)
        std = np.sqrt(Dx+eps) # (1,D) # добавление eps =1e-5
        inverted_std = 1/std # (1,D) добавление сюда eps = 1e-5 дает ошибку dx  порядка 1e-5 
        x_normed = x_centered*inverted_std # (N,D)
        out = gamma*x_normed + beta 

        # для вычисления прямого прохода layer norm можно исп код написанный для batch norm
        if (not bn_param.get("layer_norm", False)): # для батч нормализации
          # эти средние и дисперсии будут использованы во время инференса
          running_mean = momentum*running_mean + (1-momentum)*mx # (1,D)
          running_var = momentum*running_var + (1-momentum)*Dx # (1,D)
          cache = (gamma,x_normed,inverted_std,x_centered,std,Dx,eps, "batch_norm")
        else: # для layer norm
          cache = (gamma,x_normed,inverted_std,x_centered,std,Dx,eps, "layer_norm")

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_normed = (x-running_mean)/(np.sqrt(running_var+eps))
        out = gamma*x_normed + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    gamma,x_normed,inverted_std,x_centered,std,Dx,eps, norm_type = cache
    N,D = x_centered.shape

    # используем уже написанный код для вычисления градиентов layer norm
    axis = 0 # если "batch_norm"
    if norm_type == "layer_norm": 
      axis = 1 # используется только для dgamma, dbeta
      '''
      layer_norm: dout (D,N), x_normed (D,N), dgamma, dbeta (1,D)
      '''
    # сумма так как у нас градиент вычисляется по батчу (нескольким экземплярам данных) => сумма градиентов
    dgamma = np.sum(x_normed*dout,axis = axis) # sum((N,D) * (N,D),axis)=(1,D) 
    dx_normed = dout*gamma # = (N,D)
    dbeta = np.sum(dout,axis = axis)  # =(1,D)

    dx_centered = dx_normed*inverted_std # =(N,D)
    dinverted_std = np.sum(dx_normed*x_centered, axis = 0) # = (1,D)
    dstd = dinverted_std*(-1)*std**(-2)
    dDx = dstd*0.5/np.sqrt(Dx + eps) # так как eps добавляли под корень, eps нужно взять тот что был при forward pass
    dx_centered_sq = dDx/N*np.ones((N,D)) # =(N,D)
    
    dx_centered_2 = dx_centered_sq*2*x_centered # (N,D)
    dx_centered += dx_centered_2
    dx = dx_centered*1
    dmx = (-1)*np.sum(dx_centered, axis = 0) # (1,D) 
    dx_2 = dmx/N*np.ones((N,D)) # (N,D) 
    dx+= dx_2
    ## умножение на np.ones((N,D)) по сути вычислений ничего не меняет , это сделано для контроля размерности, можно и обойтись этим
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    gamma,x_normed,inverted_std,x_centered,std,_,_,norm_type = cache
    N,D = x_centered.shape

    # используем уже написанный код для вычисления градиентов layer norm
    axis = 0 # "batch_norm"
    if norm_type == "layer_norm":
      axis = 1
      '''
      для layer_norm: dout (D,N), x_normed (D,N), dgamma, dbeta (1,D)
      '''
    dgamma = np.sum(dout*x_normed,axis = axis) # (1,D) // для layer norm тоже (1,D), dout*x_normed (D,N)
    dbeta = np.sum(dout, axis = axis) # (1,D) // для layer norm тоже (1,D)
    dL_dx_normed = dout*gamma # (N,D) // для layer norm подается dout.T (D,N), в кэше gamma, beta  (1,D) уже в виде вектора столца (D,1)

    # Dx - дисперсия, mx - мат. ожидание
    dL_dDx = np.sum(dL_dx_normed*x_centered, axis = 0)*-0.5*inverted_std**3 # тут такая подстановка  std**(-3) = inverted_std**3 чтоб не пересчиытвать
    # dL/dmx = ([dL/dDx*dDx/dmx] + [dL/dxnorm*dxnorm/dmx]), dL/dDx dDx/dmx = 0 
    dL_dmx =  np.sum(-dL_dx_normed/std, axis = 0)  # (1,D)
    #dL/dx = dL/dDx * dDx/dx + dL/dmx * dmx/dx + dL/dx_norm * dxnorm/dx * dmx/dx
    dx =  dL_dDx*2*x_centered/N + dL_dmx/N + inverted_std*dout*gamma # (N,D)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    '''
    x = x.T #(D,N)
    mean_features = np.mean(x, axis = 0) # (1,N)
    f_centered = (x - mean_features) # (D,N)
    variance_features = np.mean(f_centered**2, axis = 0) # (1,N)
    std = np.sqrt(variance_features + eps) # (1, N)
    inverted_std = 1/std
    features_normalized = (x - mean_features)*inverted_std
    out = gamma.reshape((-1,1)) * features_normalized + beta.reshape((-1,1))
    out = out.T
    '''
    # используем код уже написанный для batch norm
    ln_param["mode"] = "train"
    ln_param["layer_norm"] = True
    out, cache = batchnorm_forward(x.T, gamma.reshape((-1,1)), beta.reshape((-1,1)),ln_param)
    out = out.T

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # так же используем код написанный для вычисления градиентов batch norm
    # dgamma и dbeta в вычислениях должны размером (1,D): для каждого нейрона (признака) свои gamma и beta
    dx, dgamma, dbeta = batchnorm_backward_alt(dout.T, cache) # dout.T (D,N)
    dx = dx.T

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Выборка из равномерного распределения значений от 0 до 1
        mask = (np.random.rand(*x.shape) < p) / p # inverted dropout
        out = x*mask
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x
       
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # обучаются только те нейроны выход которых был !=0, и домножаем на 1/p
        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    '''
    создадим массив заполненый нулями с размером вывода слоя сертки. Значения этого массива будут
    изменены вычисленые значения в результате сверток рецептивных полей и весов.
    '''
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    pad = conv_param["pad"]
    stride = conv_param["stride"]

    H_new = int( 1 + (H + 2*pad - HH ) / stride)
    W_new = int( 1 + (W + 2*pad  - WW) / stride)
    out = np.zeros((N, F, H_new, W_new))

    n = 0
    while n < N:
      f = 0 
      while f < F:
        '''
        Вычисляем карту активации применяя текущий фильтр.
        Нужно вычислить скалярное произведение + bias отсчетов рецептивного поля во входном массиве и весов фильтра. Так по всему пространству ширины, высоты
        используя веса текущего фильтра (так как применяется совместое использование весов нейронов одного среза глубины)
        '''
        h_new_i = 0 # используем индекс по уже вычисленному размеру высоты и ширины массива значений выходного слоя (кол-во нейронов по высоте и ширине)
        x_pad = np.pad(x[n,:,:,:], ((0,0), (pad, pad), (pad, pad)), constant_values = 0) # по оси с кол-вом фильтров не добавляем значения (в этом случае массивы)
        while h_new_i < H_new:
          w_new_i = 0
          while w_new_i < W_new:
            out[n,f,h_new_i, w_new_i] = np.sum(x_pad[:, h_new_i*stride : h_new_i*stride + HH, w_new_i*stride :  w_new_i*stride + WW] * w[f]) + b[f]
            w_new_i +=1
          h_new_i +=1
        f+=1
      n+=1

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, w, b, conv_param = cache
    
    '''
    Нужно вычислить dL/dw, dL/dx:  частные производные функции ошибки по точкам данных
    и частные производные  по весам фильтров (нейронов) сверточного слоя
      dL/dx = dL/dout * dout/dx;
      dL/dw = dL/dout * dout/dw
      dL/db = dL/dout * dout/db.
    dL/dout - восходящие градиенты даны в массиве dout (N,F,H,W)
    локальные градиенты:
      dout/dx (N,C,H_pad,W_pad)
      dout/dw (F,C,HH,WW)
      dout/db (F)
    Рассмотрим локальные градиенты где x,w - вектора, b - скаляр:
      dout/dx = d(<x,w>+b)/dx = w
      dout/dw = d(<x,w>+b)/dw = x
      dout/db = d(<x,w>+b)/db = 1
      
      Так как нейроны сверточного слоя на одной глубине массива нейронов сверточного слоя используют одинаковые веса (w,b),
      то градиенты dw будут вычислены для каждого нейрона со своим рецептивным полем на одной глубине и проссумированы.
      Также суммарные градиенты весов dw для каждой глубины F будут проссумированы по экземплярам в батче.
      Аналогично с градиентами по db: суммируем по каждому рецептивному полю одной глубины из F, затем по каждому экземпляру данных в батче.
      
      Точки входных данных одного и того же рецептивного поля исользуются каждым нейронои по глубине F. К тому же рецептивные
      поля нейронов расположенных рядом по высоте,ширине погут пересекаться. Поэтому суммирование градиентов по данным dx
      будет производиться полностью по глубине F и частично локально по ширине,высоте (если там будет пересечение рецептив. полей).

      Также следует учесть, что входом сверточного слоя был массив x в котором границы дополнены нулями (padding).
      И массив локальных градиентов dout/dx будте размером (N,С, H+2*pad, W+2*pad) и высчитав градиент dL/dx следует 
      обрезать массив до (N,C,H,W)

    '''

    N, F, Hnew_pad, Wnew_pad = dout.shape # формула размера выхода слоя по одной оси: Hnew = (H + 2*pad - HH)/stride  + 1
    N, C, H, W = x.shape 
    F, C, HH, WW = w.shape
    pad = conv_param["pad"]
    stride = conv_param["stride"]

    dw = np.zeros((F, C, HH, WW))
    dx = np.zeros((N,C,H,W))

    db = np.sum(dout, axis = (0,2,3))

    n = 0
    while n < N:
      f = 0
      dx_buff = np.zeros((C,H+2*pad,W+2*pad))
      while f < F:
        x_pad = np.pad(x,((0,0), (0,0), (pad,pad), (pad,pad)))
        h_new_i = 0
        dw_buff_for_depth_slice = np.zeros((C, HH, WW)) # тут хранится и обновляется сумма градиентов (локальные * восходящие) весов dw по карте активации
        while h_new_i < Hnew_pad:
          w_new_i = 0
          while w_new_i < Wnew_pad:
            dw_buff_for_depth_slice += x_pad[n,:,h_new_i*stride: h_new_i*stride + HH,
                            w_new_i*stride: w_new_i*stride + WW] * dout[n,f,h_new_i, w_new_i] # (C, HH, WW) * 1

            dx_buff[:,h_new_i*stride: h_new_i*stride + HH,
                      w_new_i*stride: w_new_i*stride + WW] += w[f,:,:,:] * dout[n,f,h_new_i,w_new_i] # (C, HH, WW) * 1
            w_new_i +=1
          h_new_i +=1
        dw[f] += dw_buff_for_depth_slice # += так как еще суммирование градиентов по экземплярам в батче
        f+=1
        dx[n]= dx_buff[:,pad:-pad,pad:-pad] 
      n+=1
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N,C,H,W = x.shape
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']
    H_new, W_new = int((1 + (H - pool_h)/stride)), int((1 + (W - pool_w)/stride))
    out = np.zeros((N,C, H_new, W_new))
    
    '''
    n = 0
    while n < N:
      c = 0
      while c < C:
        hi = 0
        while hi < H_new:
          wi = 0 
          while wi < W_new:
            out[n,c,hi,wi] = np.max(x[n,c,hi*stride : hi*stride + pool_h,
                                          wi*stride : wi*stride + pool_w])
            wi+=1
          hi+=1
        c+=1
      n+=1
    '''
    hi = 0
    while hi < H_new:
      wi = 0 
      while wi < W_new:
        out[:,:,hi,wi] = np.max(x[:,:,hi*stride : hi*stride + pool_h,
                                          wi*stride : wi*stride + pool_w], axis = (-1,-2))
        wi+=1
      hi+=1

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    '''
    dL/dx = dL/dout * dout/dx
    dout/dx = 1 , если этот xi максимум по окну
            = 0 иначе
    обратное распространение для операции max(x,y) означет то что градиент проходит только для числа 
    которое было максимальным из x,y, для остальных производная dL/dxi = 0

    '''
    x, pool_param = cache
    N,C,H,W = x.shape
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']
    
    H_pool,W_pool = int((1 + (H - pool_h)/stride)), int((1 + (W - pool_w)/stride))


    dx = np.zeros(x.shape)

    n = 0
    while n < N:
      c = 0
      while c < C:
        hi = 0
        while hi < H_pool:
          wi = 0 
          while wi < W_pool:
            pool_arr = x[n,c,hi*stride : hi*stride + pool_h, wi*stride : wi*stride + pool_w]
            max_index = np.unravel_index(pool_arr.argmax(), pool_arr.shape) # (i,j)
            dx[n,c,hi*stride + max_index[0], wi*stride + max_index[1]] = dout[n,c,hi,wi]
            wi+=1
          hi+=1
        c+=1
      n+=1
 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    '''
    Особенностью sparial batch norm является то что вычислияем статистики (мат ожид, дисперсию) распределений 
    для каждого выхода нейрона предыдущего слоя
    (для каждого входного значения для нейронов текущ слоя) вычисляются
    по пакету + значениям одного "среза глубины" (карты активации), так как нейроны одного среза глубины используют 
    одни и те же веса.
  
    чтобы использовать уже написанный ранее код для обычной batch normalization (вход (N,D))
    нужно преобразовать форму входного массива (N,C,H,W) к (N*H*W, C),  
    а результат затем снова к форме (N,C,H,W)
    '''
    N,C,H,W = x.shape
    x = np.transpose(x, (1,0,2,3)).reshape(C, -1) # (N,C,H,W) -> (C,N,H,W) -> (C,N*H*W)
    x = np.transpose(x) # (C,N*H*W)-> (N*H*W,C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param) # out (N*H*W,C)
    out = np.transpose(out) # (N*H*W,C) -> (C,N*H*W)
    out = out.reshape(C,N,H,W)
    out = np.transpose(out,(1,0,2,3))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W = dout.shape
    # (N,C,H,W)->(C,N,H,W)->(C,N*H*W)
    dout = np.transpose(dout, (1,0,2,3)).reshape(C,-1)
    # (C,H*W*N)->(H*W*N,C)
    dout = np.transpose(dout)
    dx, dgamma, dbeta = batchnorm_backward(dout, cache) # dx (H*W*N,C)
    dx = np.transpose(np.transpose(dx).reshape(C,N,H,W), (1,0,2,3))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    '''
    статистики для нормализации вычисляются по одному экземпляру данных, по всей ширине, высоте,
    по глубине согласно разбиению на группы. То есть для каждой группы в общем случае свое кол-во каналов
    
    G: целое число груп. должен быть делителем C по условию
    gamma,beta : (C,)
    '''
    N,C,H,W = x.shape
    # C/G - кол-во каналов в группе
    C_per_G = int(C/G) # по условию G - делитель C, поэтому не будет явного округления.
    
    # (N,C,H,W) -> (нужно преобразовать в двумерный массив чтоб выполнять операции для нормализации)
    # (N,C,H,W) -> (N*G, C/G * H * W) всего N*G групп (для разных экземпляров данных разные группы)
    x = x.reshape(N*G,C_per_G*H*W) 
    # для вычисления разности двух массивов изменим форму x
    x = x.T # (C/G*H*W, N*G), C/G*H*W - кол-во признаков num_fts, N*G - обшее кол-во групп по всем экземплярам
    mx = np.mean(x, axis = 0) # (N*G)
    x_centered = x-mx # (num_fts, N*G) num_fts - кол-во признаков в группе = C/G*H*W
    centered_x_sq = x_centered**2
    var = np.mean(centered_x_sq, axis = 0) # (N*G)
    std = np.sqrt(var + eps)
    inverted_std = 1/std # (N*G)
    x_normed = x_centered*inverted_std # (num_fts, N*G)
    
    x_normed = np.reshape(x_normed.T, (N,C,H,W)) # (N,C,H,W)
    gamma = gamma.reshape(1,C,1,1)
    beta = beta.reshape(1,C,1,1)

    out = gamma*x_normed + beta
    cache = (gamma,x_normed,inverted_std,x_centered,std,var,eps, C_per_G, G) # добавляем в кэш инфу о кол-ве каналов в группе и кол-ве групп

    # Group Normalization не зависит как и layer norm от типа процесса использования: train/test

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,) !!! (1,C,1,1)
    - dbeta: Gradient with respect to shift parameter, of shape (C,) !!! (1,C,1,1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W = dout.shape
    gamma,x_normed,inverted_std,x_centered,std,var,eps,C_per_G,G = cache

    # для dbeta, dgamma градиенты суммируются по размеру батча (так как обучаемся по нескольким экземплярам)
    # так и по H,W так как влияют на выход каждого нейрона одного среза глубины
    dbeta = np.sum(dout, axis = (0,2,3), keepdims=True) # (1,C,1,1) ! размер не соответствует описанию выше : (C,). Иначе не проходит проверку
    # локальный градиент по dgamma : x_normed (N,C,H,W)
    dgamma = np.sum(dout*x_normed, axis = (0,2,3), keepdims=True) # поэлементное умножение тензоров и сумма -> (1,C,1,1)

    dx_normed = dout*gamma.reshape(1,C,1,1) # поэлементное умножение  (N,C,H,W)
    # преобразование в форму которая использовалась в прямом проходе для x_normed
    dx_normed = np.reshape( dx_normed , (N*G, C_per_G * H * W)).T # (num_fts, N*G)
   
    '''
    Тут дальше можно реализовать обратное распространение на основе вычисления прямого прохода в функции spatial_groupnorm_forward.
    Либо использовать уже упрощеный написанный код для прямого вычисления dx по выведенной формуле см batch_norm_backward_alt для условия layer_norm,
    внеся некоторые изменения
    '''
    dL_dx_normed = dx_normed
    # изменения относительно batch_norm_backward_alt для условия layer_norm 
    # [
    dout_mul_gamma = dout*gamma
    dout_mul_gamma = np.reshape( dout_mul_gamma , (N*G, C_per_G * W * H)).T # (N,C,H,W) -> (N*G, C_per_G*H*W) = (N*G,num_fts) -> (num_fts, N*G)
    C_per_G_W_H = C_per_G*W*H
    # ]

    # Dx - дисперсия, mx - мат. ожидание
    dL_dDx = np.sum(dL_dx_normed*x_centered, axis = 0)*-0.5*inverted_std**3  # (N*G)
    dL_dmx =  np.sum(-dL_dx_normed/std, axis = 0)  # (num_gr)
    dx =  dL_dDx*2*x_centered/C_per_G_W_H + dL_dmx/C_per_G_W_H + inverted_std*dout_mul_gamma # (num_fts, N*G)
    
    dx = np.reshape(dx.T, (N,C,H,W)) # (num_fts, N*G) -> (N*G,num_fts) = (N*G, C_per_G * H * W) -> (N,С,H,W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
