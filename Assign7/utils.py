# utils.py
# Author: Skyler Ehly
# 2-1-2018

import support
import numpy as np
import math

# Slides the given kernel/mask over the image and returns the result.
# The result is smaller than the input image
def convolve(image, kernel):
    #image_matrix = support.loadImage(image)
    image_matrix = image
    image_rows, image_cols = support.getSize(image_matrix)
    kernel_rows = support.getRows(kernel)
    new_image_rows = image_rows - kernel_rows + 1
    new_image_cols = image_cols - kernel_rows + 1
    # A new matrix of zeros of final size of new image
    new_image = support.makeMatrix(new_image_rows, new_image_cols)
    # for each row in new image
    for r in range(0, new_image_rows, 1):
        end_row = r+kernel_rows
        # for each col in new image
        for c in range(0, new_image_cols, 1):
            # current matrix of image values
            end_col = c+kernel_rows
            conveq_matrix = image_matrix[r:end_row, c:end_col]
            # convolve image values and kernel
            new_image[r,c] = conveq(conveq_matrix, kernel)
    return new_image

# Convolves two matrices of the same dimensions
# Multiplies element-wise and sums the entries in the result
def conveq(M1, M2):
    return np.sum(M1*M2)

# Applies the given directional kernel to the given image and returns:
#   the gradients in the x and y directions
#   the gradient magnitude computed as sqrt(Gx^2+Gy^2) for each pixel
#   the direction of the gradient computed as atan(Gy/Gx) for each pixel
# Gx, Gy, G, D
def getEdges(image, kernelX, kernelY):
    Gx = convolve(image, kernelX)
    Gy = convolve(image, kernelY)
    G = np.sqrt(Gx*Gx + Gy*Gy)
    D = np.arctan2(Gy,Gx)
    return Gx, Gy, G, D

# Applies the given smoothing filter to the given image
def smoothImage(image, kernel):
    return convolve(image, kernel)

# Returns the kernel specified by the given paramaters
# Possible params
#   "prewitt"
#     returns the pair (Kx, Ky) represening Prewitt's operator for both  
#     directions
#   "sobel"
#     returns the pair (Kx, Ky) representing Sobel's operator for both
#     directions
#   "average", n
#     returns the nXn averageing kernel (all cells have the same values
#     and add up to 1)
#   "gauss", n, sigma
#     returns the nXn Gaussian kernel with the given std. deviation sigma;
#     the values in the kernel are computed based on the formula from Wikipedia;
#     x, y range from [-n/2..n/2];
#     before returning the kernel normalize the values, so that the total sum is 1
def makeKernel(*params):
    if (params[0] == 'prewitt'):
	Kx = support.makeMatrix([(-1, 0, 1), (-1, 0, 1), (-1, 0, -1)] )
	Ky = support.makeMatrix([(-1, -1, -1), (0, 0, 0), (1, 1, 1)])
	return Kx, Ky
    elif (params[0] == 'sobel'):
	Kx = support.makeMatrix([(-1, 0, 1), (-2, 0, 2), (-1, 0, -1)] )
	Ky = support.makeMatrix([(-1, -2, -1), (0, 0, 0), (1, 2, 1)])
	return Kx, Ky
    elif (params[0] == 'average'):
	n = params[1]
	value = 1.0/(n*n)
	K = support.makeMatrix(n, n, value)
	return K
    elif (params[0] == 'gauss'):
	n = params[1]
	sig = params[2]
	K = support.makeMatrix(n,n)
        bound = n/2
        r = 0
        for y in range(-bound,bound+1,1):
            c = 0
            for x in range(-bound,bound+1,1):
                value = 1/(2*math.pi*(sig*sig)) * math.exp(-(x*x+y*y)/(2*sig*sig))
                K[r,c] = value
                c = c+1
            r = r+1

        # Normalize array
        K = K / np.sum(K)
        return K

# returns the gassian matrix 	
def computeGauss(n, sig):
    K = support.makeMatrix(n,n)
    bound = n/2
    r = 0
    for y in range(-bound,bound+1,1):
        c = 0
        for x in range(-bound,bound+1,1):
            value = 1/(2*math.pi*(sig*sig)) * math.exp(-(x*x+y*y)/(2*sig*sig))
            K[r,c] = value
            c = c+1
        r = r+1

    # Normalize array
    K = K / np.sum(K)
    return K
