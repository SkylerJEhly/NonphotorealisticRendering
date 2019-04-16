"""
File: support.py

This file contains a collection of functions for
saving and loading from/to numpy based matrices.

1. Matrices -- The returned matrix is a numpy 2D array,
               and all operations from numpy can be used.

# creating
    M = makeMatrix(5, 7)        # matrix of 5 rows, 7 columns, init to 0
    M = makeMatrix(5, 7, 3)     # matrix of 5 rows, 7 columns, init to 3
    M = makeMatrix([[1, 2],     # matrix from Python list
                    [3, 4]])

# dimensions
    rows, cols = getSize(M)
    rows = getRows(M)
    cols = getCols(M)

# indexing
    M[1, 2] = 10
    value = M[1, 2]

# slicing
    A = M[1, 2:5]      # 1x3 matrix: row 1, columns 2,3,4
    B = M[2:5, 1]      # 3x1 matrix: rows 2,3,4, column 1
    C = M[2:5, 1:3]    # 3x2 matrix: rows 2,3,4, columns 1,2

# operations
    A = 255*M        # multiply each element by 255
    B = 255 + M      # add 255 to each element
    
    C = M1 + M2      # add the matrices (element by element)
    D = M1 * M2      # [not matrix multiplication] 
                     # multiply the matrices (element by element)

    value = numpy.sum(M*M)    # multiply element-wise and sum the result

2. Images

    M = loadImage("glatfleter.png")    # load an image into numpy 2D array
    M /= 2.0                           # scale down the colors
    saveImage(M, "result.png")         # save the result
"""


import numpy
import Image


# creates a matrix
def makeMatrix( *args ):
    if len(args) == 1:
        # from python list
        return numpy.array( args[0] )
    elif len(args) == 2:
        # from (rows, cols)
        return numpy.zeros( (args[0], args[1]) )
    elif len(args) == 3:
        # from (rows, cols, value)
        return args[2]*numpy.ones( (args[0], args[1]) )

# returns the number of rows of the given matrix
def getRows(M):
    return M.shape[0]

# returns the number of columns of the given matrix
def getCols(M):
    return M.shape[1]

# returns the dimensions (rows, cols) of the given matrix
def getSize(M):
    return getRows(M), getCols(M)


# returns a numpy 2D array from the image in the given file
def loadImage(filename):
    image = Image.open(filename)
    pixMap = image.load()
    cols, rows = image.size[0], image.size[1]   # width{0}=>cols,height{1}=>rows
    pixels = makeMatrix(rows, cols)
    for r in range(rows):
        for c in range(cols):
            pixels[r, c] = pixMap[c, r]         # (x,y)=(c, r)=>(r, c)
    return pixels

# saves the given 2D array as an image with the given filename in PNG format
# (the .png extension is added to _filename_, if not present)
def saveImage(M, filename):
    # convert to [0..255]
    M = M - numpy.min(M)
    M = 255*M/numpy.max(M)

    # save the image
    rows, cols = getRows(M), getCols(M)
    image = Image.new("L", (cols, rows))        # cols=>width,rows=>height
    pixels = image.load()
    for r in range(0, rows):
        for c in range(0, cols):
            pixels[c, r] = int(M[r, c])         # (r,c)=(y,x)=>(x, y)
    if not filename.endswith(".png"):
        filename += ".png"
    image.save(filename, "png")