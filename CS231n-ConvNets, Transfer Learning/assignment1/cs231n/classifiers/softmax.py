import numpy as np
from random import shuffle

def softmax(W, x, k):
    normalizer = W[np.argmax(W.dot(x))]
    numerator = np.exp( (W[k] - normalizer).dot(x) )
    denominator = np.sum( np.exp( (W - normalizer).dot(x)) )
    return numerator/denominator

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[1]
  num_classes = W.shape[0]

  for n in xrange(X.shape[1]):
    x = X[:, n]
    softmax_for_example = softmax(W, x, y[n])
    loss -= np.log( softmax_for_example )
    for c in xrange(num_classes):
      if c != y[n]:
        dW[c, :] += softmax(W, x, c) * x.T
      else:
        dW[c,:] += (softmax_for_example - 1) * x.T

  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W ** 2)
  dW += reg * W




  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[1]
  num_classes = W.shape[0]

  Y = np.eye(num_classes)[:,y]  # C x N matrix; 1-of-k column encoding for each example
  Numer_no_exp = W.dot(X)  # column j gives (unexponentiated) numerators of softmaxes for example j
  normalizer = np.max(Numer_no_exp, 0)  # max of each column
  Numer = np.exp(Numer_no_exp - normalizer)  # column j gives the normalized softmax numerators for example j
  Softmaxes = Numer / np.sum(Numer, axis = 0)  # sum -> softmax denominator
  # Y keeps the only right softmax/col; sum squashes softmaxes to a row (there's one softmax/column)
  loss = np.sum(Y * Softmaxes, axis = 0)
  loss = np.sum(-1 * np.log(loss))

  dW = (Softmaxes - Y).dot(X.T)

  loss /= num_train
  loss += 0.5 * reg * np.sum(W ** 2)
  dW /= num_train
  dW += reg * W

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
