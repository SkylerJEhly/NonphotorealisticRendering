"""
File: support.py

This file contains a collection of functions for
saving and loading from/to numpy based matrices.

1. Matrices

    M = makeMatrix(5, 7)    # creates a matrix with 5 rows and 7 columns

    rows = getRows(M)       # finds the number of rows of the matrix
    cols = getCols(M)       # finds the number of columns of the matrix

    M[1, 2] = 10            # stores the value 10 in cell (1, 2)
    value = M[1, 2]         # copies the contents of cell (1, 2) into a variable

    print M                 # shows the matrix on the screen

    The returned matrix is a numpy 2D array,
    and all operations from numpy can be used.

2. Images

    M = loadImage( "glatfleter.png" )    # load an image into numpy 2D array
    M /= 2.0                             # scale down the colors
    saveImage(M, "result.png")           # save the result

    showImage(M)                         # shows the image in separate window
"""


import numpy
import Image


# creates a matrix from:
# (a) python list -- makeMatrix( [[1, 2, 3], [4, 5, 6]] )
# (b) given (row, col) dimensions -- makeMatrix( 2, 3 )
# (c) given (row, col) dimensions and defualt value -- makeMatrix( 2, 3, 255 )
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


# creates a vector from:
# (a) python list -- makeVector( [1, 2] )
# (b) given (x, y) coords -- makeVector( 1, 2 )
def makeVector( *args ):
    if len(args) == 2:
        # from pair of coords: makeVector(1, 2)
        return makeMatrix( args )
    else:
        # from list or vector: makeVector( [1, 2] ), makeVector( z )
        coords = args[0]
        return makeMatrix( [coords[0], coords[1]] )


# returns a numpy 2D array from the image in the given file
def loadImage(filename):
    image = Image.open(filename)
    pixMap = image.load()
    cols, rows = image.size[0], image.size[1]   # width(0)=>cols,height(1)=>rows
    pixels = makeMatrix(rows, cols)
    for r in range(rows):
        for c in range(cols):
            pixels[r, c] = pixMap[c, r]         # (x,y)=(c, r)=>(r, c)
    return pixels

# saves the given 2D array as an image with the given filename in PNG format
# (the .png extension is added to _filename_, if not present)
def saveImage(image, filename):
    if not filename.endswith(".png"):
        filename += ".png"
    try:
        image.save(filename, "png")
    except:
        # remap to 0..255
        image = image - numpy.min(image)
        if numpy.max(image) != 0:
            image = image/numpy.max(image)
            image = 255*image
        
        image = mat2img( image )
        image.save(filename, "png")

# creates an image with the given dimensions with all
# pixels set to the given value (default white)
def makeImage(rows, cols, value=255):
    image = Image.new("L", (cols, rows))        # cols=>width,rows=>height
    pixels = image.load()
    for r in range(0, rows):
        for c in range(0, cols):
            pixels[c, r] = value                # (r,c)=(y,x)=>(x, y)
    return image

# converts the given matrix to PIL image
def mat2img( M ):
    rows, cols = getSize( M )
    image = makeImage( rows, cols, 0 )
    pixels = image.load()
    for r in range(0, rows):
        for c in range(0, cols):
            pixels[c, r] = int(M[r, c])              # (r,c)=(y,x)=>(x, y)
    return image

# shows a window with the image (either a matrix or PIL image)
def showImage( image ):
    try:
        # if PIL image
        image.show()
    except:
        # if Numpy matrix
        # remap to 0..255
        image = image - numpy.min(image)
        if numpy.max(image) != 0:
            image = image/numpy.max(image)
            image = 255*image
        
        image = mat2img( image )
        image.show()
