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

    num_examples = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_examples):
        scorei = X[i].dot(W)
        exp_scorei = np.exp(scorei)
        probs = exp_scorei / np.sum(exp_scorei)

        loss += -np.log(probs[y[i]])
        dscore = probs
        dscore[y[i]] -= 1
        for k in range(num_classes):
            dW[:,k] += np.dot(dscore[k],X[i])

    loss /= num_examples
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_examples
    dW += reg*W


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_examples = X.shape[0]

    # evalueate class scores.Note that here b is embadded into scores
    scores = np.dot(X,W)

    # compute class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # compute the losses: average cross-entropy loss and regularization
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs) / num_examples
    reg_loss = 0.5 * reg * np.sum(W*W)
    loss = data_loss + reg_loss

    # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    # backpropagate the gradient to the parameters
    dW = np.dot(X.T, dscores)
    dW += reg*W

    return loss, dW

