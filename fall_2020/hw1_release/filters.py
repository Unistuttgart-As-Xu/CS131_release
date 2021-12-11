"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    
    ### YOUR CODE HERE
    kernel_fliped = np.zeros_like(kernel)
    for i in range(Hk):
        for j in range(Wk):
            kernel_fliped[i,j] = kernel[Hk-1-i, Wk-1-j]

    for i in range(Hi):
        for j in range(Wi):
            sum_s = 0
            start_i = i - int(Hk/2.0)
            start_j = j - int(Wk/2.0)
            for ii in range(Hk):
                for jj in range(Wk):
                    if start_i+ii < 0 or start_j+jj < 0 or start_i + ii >= Hi or start_j + jj >= Wi:
                        continue
                    else:
                        sum_s = sum_s + kernel_fliped[ii,jj] * image[ii + start_i, jj + start_j]
            out[i,j] = sum_s
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    H_out = H + pad_height * 2
    W_out = W + pad_width * 2
    out = np.zeros((H_out, W_out))
    out[pad_height:-pad_height, pad_width:-pad_width] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    image = zero_pad(image, int(Hk/2.0), int(Wk/2.0))
    kernel = np.flip(kernel, 0)
    kernel = np.flip(kernel, 1)

    for i in range(Hi):
        for j in range(Wi):
            out[i,j] = np.sum(kernel * image[i:i + Hk, j:j + Wk])
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g = np.flip(g, 0)
    g = np.flip(g, 1)
    out = conv_fast(f, g)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g -= np.mean(g)
    out = cross_correlation(f, g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    out = np.zeros_like(f)
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    image = zero_pad(f, int(Hk/2.0), int(Wk/2.0))
    g = (g - np.mean(g)) / np.std(g)
    for i in range(Hi):
        for j in range(Wi):
            img_patch = image[i:i+Hk, j:j+Wk]
            img_patch = (img_patch - np.mean(img_patch)) / np.std(img_patch)
            out[i,j] = np.sum(g * img_patch)
    ### END YOUR CODE

    return out
