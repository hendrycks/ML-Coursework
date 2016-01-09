import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]

  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]

    for c in xrange(num_classes):

      if c == y[i]:
        for j in xrange(num_classes):
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0 and c != j:
              dW[c,:] -= X[:, i].T
      else:
        margin = scores[c] - correct_class_score + 1 # note delta = 1
        if margin > 0:
          loss += margin
          dW[c,:] += X[:, i].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
            

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_classes = W.shape[0]
  num_train = X.shape[1]

  # Column j has 1 only in the y^j spot
  Example_classes = np.eye(num_classes)[:,y]

  # W[y[...]].T part makes row class x^i be column
  Passing_examples = W.dot(X) - np.sum(W[y[range(num_train)]].T * X, axis = 0) + np.ones(num_train) - Example_classes

  Indicators_from_max = Passing_examples > 0

  # The filter is > -1 perform the max operation; the j = yi case is 0 so it doesn't matter
  loss = np.sum(  Passing_examples * Indicators_from_max  )

  loss /= num_train
  loss += 0.5 * reg * np.sum(W ** 2)


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  Grad_weights = Indicators_from_max - Example_classes * np.sum(Indicators_from_max, axis = 0)
  dW = Grad_weights.dot(X.T)

  dW /= num_train
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
