import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def crop_image(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index we want to include in our cropped image.
        start_col (int): The starting column index we want to include in our cropped image.
        num_rows (int): Number of rows in our desired cropped image.
        num_cols (int): Number of columns in our desired cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = image.copy()
    out = out[start_row:(start_row+num_rows), start_col:(start_col+num_cols), :]
    ### END YOUR CODE

    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = image.copy()
    out = out**2*0.5
    ### END YOUR CODE

    return out


def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))

    # 2. Populate the `output_image` array using values from `input_image`
    #    > This should require two nested for loops!

    ### YOUR CODE HERE
    row_scale_factor = float(input_image.shape[0])/(output_rows)
    col_scale_factor = float(input_image.shape[1])/(output_cols)
    for i in range(0, output_rows):
        for j in range(0, output_cols):
           output_image[i,j,:] = input_image[int(i*row_scale_factor),int(j*col_scale_factor),:]

    ### END YOUR CODE

    # 3. Return the output image
    return output_image


def rotate2d(point, theta):
    """Rotate a 2D coordinate by some angle theta.

    Args:
        point (np.ndarray): A 1D NumPy array containing two values: an x and y coordinate.
        theta (float): An theta to rotate by, in radians.

    Returns:
        np.ndarray: A 1D NumPy array containing your rotated x and y values.
    """
    assert point.shape == (2,)
    assert isinstance(theta, float)

    # Reminder: np.cos() and np.sin() will be useful here!

    ## YOUR CODE HERE
    out = point.copy()
    new_out = np.zeros(shape=(2), dtype=np.float)
    new_out[0] = out[0]*np.cos(-theta) + out[1]*np.sin(-theta)
    new_out[1] = -out[0]*np.sin(-theta) + out[1]*np.cos(-theta)
    return new_out
    ### END YOUR CODE


def rotate_image(input_image, theta):
    """Rotate an image by some angle theta.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        theta (float): Angle to rotate our image by, in radians.

    Returns:
        (np.ndarray): Rotated image, with the same shape as the input.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create an output image with the same shape as the input
    output_image = np.zeros_like(input_image)

    ## YOUR CODE HERE
    for i in range(0, input_image.shape[0]):
        for j in range(0, input_image.shape[1]):
           point = np.array([i,j])
           point[0] = point[0] - input_rows/2.0
           point[1] = -point[1] + input_cols/2.0
           new_point = rotate2d(point, -theta)
           new_point[0] = new_point[0] + input_rows/2.0
           new_point[1] = -new_point[1] + input_rows/2.0
           if (int(new_point[0]) >= input_image.shape[0] or int(new_point[1]) >= input_image.shape[1] or int(new_point[0]) < 0 or int(new_point[1]) < 0): 
                output_image[i,j,:] = 0
           else:
                output_image[i,j,:] = input_image[int(new_point[0]),int(new_point[1]),:]
    ### END YOUR CODE

    # 3. Return the output image
    return output_image
