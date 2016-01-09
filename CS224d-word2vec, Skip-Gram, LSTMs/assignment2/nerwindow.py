from numpy import *
from nn.base import NNBase
from nn.math import softmax
from misc import random_weight_matrix


##
# Evaluation code; do not change this
##
from sklearn import metrics
def full_report(y_true, y_pred, tagnames):
    cr = metrics.classification_report(y_true, y_pred,
                                       target_names=tagnames)
    print cr

def eval_performance(y_true, y_pred, tagnames):
    pre, rec, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred)
    print "=== Performance (omitting 'O' class) ==="
    print "Mean precision:  %.02f%%" % (100*sum(pre[1:] * support[1:])/sum(support[1:]))
    print "Mean recall:     %.02f%%" % (100*sum(rec[1:] * support[1:])/sum(support[1:]))
    print "Mean F1:         %.02f%%" % (100*sum(f1[1:] * support[1:])/sum(support[1:]))

##
# Implement this!
##
class WindowMLP(NNBase):
    """Single hidden layer, plus representation learning."""

    def __init__(self, wv, windowsize=3,
                 dims=[None, 100, 5],
                 reg=0.001, alpha=0.01, rseed=10):
        """
        Initialize classifier model.

        Arguments:
        wv : initial word vectors (array |V| x n)
            note that this is the transpose of the n x |V| matrix L
            described in the handout; you'll want to keep it in
            this |V| x n form for efficiency reasons, since numpy
            stores matrix rows continguously.
        windowsize : int, size of context window
        dims : dimensions of [input, hidden, output]
            input dimension can be computed from wv.shape
        reg : regularization strength (lambda)
        alpha : default learning rate
        rseed : random initialization seed
        """

        # Set regularization
        self.lreg = float(reg)
        self.alpha = alpha  # default training rate

        dims[0] = windowsize * wv.shape[1]  # input dimension
        param_dims = dict(W=(dims[1], dims[0]),
                          b1=(dims[1],),
                          U=(dims[2], dims[1]),
                          b2=(dims[2],),
                          )
        param_dims_sparse = dict(L=wv.shape)

        # initialize parameters: don't change this line
        NNBase.__init__(self, param_dims, param_dims_sparse)

        random.seed(rseed)  # be sure to seed this for repeatability!
        #### YOUR CODE HERE ####

        self.word_vec_size = wv.shape[1]
        self.windowsize = windowsize

        self.sparams.L = wv.copy()
        self.params.W = random_weight_matrix(*self.params.W.shape)
        #self.params.b1 = zeros((dims[1], 1))
        self.params.U = random_weight_matrix(*self.params.U.shape)
        #self.params.b2 = zeros((dims[2], 1))

        #elsewhere in the code, I think, the bias terms are defined

        #### END YOUR CODE ####

    def _acc_grads(self, window, label):
        """
        Accumulate gradients, given a training point
        (window, label) of the format

        window = [x_{i-1} x_{i} x_{i+1}] # three integers
        label = {0,1,2,3,4} # single int, gives class

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.U += (your gradient dJ/dU)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # this adds an update for that index
        """
        #### YOUR CODE HERE ####

        # retrieve word vectors
        x_0_loc = window[0]
        x_1_loc = window[1]
        x_2_loc = window[2]
        x_0 = self.sparams.L[x_0_loc]
        x_1 = self.sparams.L[x_1_loc]
        x_2 = self.sparams.L[x_2_loc]

        x = concatenate((x_0, x_1, x_2)).reshape((-1,1))

        z1 = dot(self.params.W, x) + self.params.b1.reshape((-1,1))
        h1 = tanh(z1)
        z2 = dot(self.params.U, h1) + self.params.b2.reshape((-1,1))
        out = softmax(z2)
        assert out.shape == (5, 1)  # 5 the number of labels

        dout = out
        dout[label] -= 1
        self.grads.U += dout.dot(h1.T) + self.params.U * self.lreg
        self.grads.b2 += dout.reshape((-1))
        dz1 = dot(self.params.U.T, dout) * (1 - h1*h1)
        self.grads.W += dot(dz1, x.T) + self.params.W * self.lreg
        self.grads.b1 += dz1.reshape((-1,))
        L_grad = dot(self.params.W.T, dz1).reshape((-1,))
        self.sgrads.L[x_0_loc] = L_grad[0:50]
        self.sgrads.L[x_1_loc] = L_grad[50:100]
        self.sgrads.L[x_2_loc] = L_grad[100:150]

        assert self.grads.U.shape == self.params.U.shape
        assert self.grads.b2.shape == self.params.b2.shape
        assert self.grads.W.shape == self.params.W.shape
        assert self.grads.b1.shape == self.params.b1.shape

        #### END YOUR CODE ####


    def predict_proba(self, windows):
        """
        Predict class probabilities.

        Should return a matrix P of probabilities,
        with each row corresponding to a row of X.

        windows = array (n x windowsize),
            each row is a window of indices
        """
        # handle singleton input by making sure we have
        # a list-of-lists
        if not hasattr(windows[0], "__iter__"):
            windows = [windows]
        if type(windows) is list:
            windows = array(windows)
        #### YOUR CODE HERE ####

        x_0_col = windows[:, 0]
        x_1_col = windows[:, 1]
        x_2_col = windows[:, 2]

        # get the window vectors arranged in a row; 0:-1 to keep it a column vector
        window_vecs = concatenate((self.sparams.L[x_0_col],
                                   self.sparams.L[x_1_col],
                                   self.sparams.L[x_2_col]), axis=1)

        h1 = tanh(dot(self.params.W, window_vecs.T) + self.params.b1.reshape((-1,1)))

        # Unfortunately, the softmax function we're given cannot manage a matrix
        # We must therefore use some loops
        # We won't re-write softmax because the one we're given is in Cython

        z2 = (dot(self.params.U, h1) + self.params.b2.reshape((-1,1))).T
        probs = zeros_like(z2)

        for i in xrange(z2.shape[0]):
            probs[i, :] = softmax(z2[i, :])

        #### END YOUR CODE ####

        return probs  # rows are output for each input


    def predict(self, windows):
        """
        Predict most likely class.
        Returns a list of predicted class indices;
        input is same as to predict_proba
        """

        #### YOUR CODE HERE ####

        probs = self.predict_proba(windows)

        c = argmax(probs, axis=1)

        #### END YOUR CODE ####
        return c  # list of predicted classes


    def compute_loss(self, windows, labels):
        """
        Compute the loss for a given dataset.
        windows = same as for predict_proba
        labels = list of class labels, for each row of windows
        """

        #### YOUR CODE HERE ####
        if type(labels) is int32:
            labels = [labels]

        probs = self.predict_proba(windows)

        # Since we might have a huge amount of windows we must loopify and
        # not use -sum(log(probs[:, labels]))

        J = 0
        for i in xrange(probs.shape[0]):
            J -= sum(log(probs[i, labels[i]]))

        J += self.lreg * (sum(self.params.U ** 2) + sum(self.params.W ** 2)) / 2.0

        #### END YOUR CODE ####
        return J