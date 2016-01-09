import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  reshapen_x = np.reshape(x,(x.shape[0],w.shape[0]))
  out = reshapen_x.dot(w) + b

  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
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

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  N = x.shape[0]; D = w.shape[0]; M = w.shape[1]

  dx = np.reshape( np.dot(dout, w.T), (x.shape) )
  dw = np.dot( np.reshape(x,(N,D)).T, dout )
  db = np.sum(dout, axis = 0)

  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
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
  out = np.maximum(0,x)
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
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
  dx = (x > 0) * dout
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  stride = conv_param['stride']
  pad = conv_param['pad']
  x_padded = np.lib.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant') # default padding constant is 0

  N, C, H ,W = x.shape
  F, C, HH, WW = w.shape
  horiz_windows = 1 + (H + 2 * pad - HH)/stride
  vertical_windows = 1 + (W + 2 * pad - WW)/stride
  out = np.zeros((N, F, horiz_windows, vertical_windows))

  for n in xrange(N):
    for f in xrange(F):
      for v in xrange(0, vertical_windows):
        for h in xrange(0, horiz_windows):
          out[n,f,h,v] = np.sum( x_padded[n, :, h*stride:h*stride + HH, v*stride:v*stride + WW] * w[f] ) + b[f]
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
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
  x, w, b, conv_param = cache
  stride = conv_param['stride']
  pad = conv_param['pad']
  x_padded = np.lib.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant') # default padding constant is 0

  dx, dw, db, dx_padded = np.zeros_like(x), np.zeros_like(w), np.zeros_like(b), np.zeros_like(x_padded)

  N, C, H ,W = x.shape
  F, C, HH, WW = w.shape
  horiz_windows = 1 + (W + 2 * pad - WW)/stride
  vertical_windows = 1 + (H + 2 * pad - HH)/stride

  for n in xrange(N):
    for f in xrange(F):
      for h in xrange(0, horiz_windows):
        for v in xrange(0, vertical_windows):
          dx_padded[n,:,h*stride:h*stride + WW, v*stride:v*stride + HH] += dout[n,f,h,v] * w[f]
          dw[f] += dout[n, f, h, v] * x_padded[n, :, h*stride:h*stride + WW, v*stride:v*stride + HH]
          db[f] += dout[n,f,h,v]
  dx = dx_padded[:,:,pad:-pad,pad:-pad]

  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  N, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  horiz_windows = 1 + (W - pool_width)/stride
  vertical_windows = 1 + (H - pool_height)/stride

  out = np.zeros((N,C,horiz_windows,vertical_windows))

  for n in xrange(N):
    for c in xrange(C):
      for h in xrange(horiz_windows):
        for v in xrange(vertical_windows):
          out[n,c,h,v] += np.max( x[n,c, h*stride:h*stride + pool_width, v*stride:v*stride + pool_height] )

  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  x = cache[0]
  N, C, H, W = x.shape
  pool_param = cache[1]
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  horiz_windows = 1 + (W - pool_width)/stride
  vertical_windows = 1 + (H - pool_height)/stride

  dx = np.zeros_like(cache[0])

  for n in xrange(N):
    for c in xrange(C):
      for h in xrange(horiz_windows):
        for v in xrange(vertical_windows):
          patch = x[n,c, h*stride:h*stride + pool_width, v*stride:v*stride + pool_height]
          indicator_pos = np.unravel_index(patch.argmax(), patch.shape)
          dx[n,c, h*stride:h*stride + pool_width, v*stride:v*stride + pool_height] = np.zeros_like(patch)
          dx[n,c, h*stride:h*stride + pool_width, v*stride:v*stride + pool_height][indicator_pos] = 1
          dx[n,c, h*stride:h*stride + pool_width, v*stride:v*stride + pool_height] *= dout[n,c,h,v]


  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
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
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

