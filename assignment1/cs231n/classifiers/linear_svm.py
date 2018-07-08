import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] += np.transpose(X[i])
                dW[:,y[i]] -= np.transpose(X[i])


    # Right now the loss is a sum over all training examples, average it.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg*W

    # TODO:
    # Compute the gradient of the loss function and store it dW.
    # Rather that first computing the loss and then computing the derivative,
    # it may be simpler to compute the derivative at the same time that the
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    delta = 1.0 # Set hyperparameter delta to 1
    num_train = X.shape[0]

    # Implement a vectorized version of the structured SVM loss, storing the result in loss.
    scores = X.dot(W)
    correct_scores = scores[np.arange(num_train), y]
    margins = np.maximum(0 ,scores - correct_scores[:, np.newaxis] + delta)
    margins[np.arange(num_train), y] = 0
    loss = (1.0/num_train*np.sum(margins)) + reg * np.sum(W*W)

    # Implement a vectorized version of the gradient for the structured SVM
    # loss, storing the result in dW.

    X_mask = np.zeros(margins.shape)
    X_mask[margins>0] = 1
    incorrect_count = np.sum(X_mask, axis=1)
    X_mask[np.arange(num_train), y] -= incorrect_count

    # X.T: D*N ,X_mask: N*C
    dW = X.T.dot(X_mask)
    dW /= num_train
    dW += reg*W

    return loss, dW
