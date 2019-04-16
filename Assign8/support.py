"""
File: support.py

This file contains a collection of functions for
saving and loading from/to numpy based matrices.

Vectors/Matrices:

u = makeVector( 1, 2 )      # makes the vector (1, 2)
v = makeVector( [3, 4] )    # makes the vector (3, 4)
v = makeVector( u )         # makes a copy of vector u

M = makeMatrix( [[1, 2, 3], [4, 5, 6]] )  # 2D array
M = makeMatrix( 2, 3 )                    # 2x3 matrix
M = makeMatrix( 2, 3, 255 )               # 2x3 matrix initialized to 255

M = makeVecMatrix( 4, 5, 2 )      # 4x5 matrix of 2D vectors (example: a matrix of locations)
M = makeVecMatrix( 4, 5, 3 )      # 4x5 matrix of 3D vectors (example: a matrix of colors)

rows, cols = getSize(M)  # get rows and columns
rows = getRows(M)        # get only the rows
cols = getCols(M)        # get only the columns

saveMatrix(M, "data")        # saves the matrix to file "data.npy"
M = loadMatrix("data.npy")   # loads the matrix from file "data.npy"

Mgray = rgb2gray( Mrgb )     # convert RGB matrix (each cell (R,G,B)) to GRAY matrix (each cell number)

Images:

M = loadImage("circle.pgm")      # loads the GRAY image "circle.pgm" into a 2D Numpy matrix (each cell a number)
M = loadImageRGB("scene.png")    # loads the RGB image "scene.png" into a 2D Numpy matrix (each cell (R,G,B) vector)

saveImage(image, "result")      # saves the given image/matrix to file "result.png"

image = makeImage( 100, 200, 125 )    # creates 100x200 GRAY PIL image (not matrix), each pixel set to 125

image = makeImageRGB( 100, 200 )    # creates 100x200 RGB image (not matrix), pixels are set to (0,0,0)

showImage( M )    # shows the given matrix/PIL image


Examples:

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
    dtype=numpy.float64
    if len(args) == 1:
        # from python list
        return numpy.array( args[0], dtype=dtype )
    elif len(args) == 2:
        # from (rows, cols)
        return numpy.zeros( (args[0], args[1]), dtype=dtype )
    elif len(args) == 3:
        # from (rows, cols, value)
        return args[2]*numpy.ones( (args[0], args[1]), dtype=dtype )

# creates a matrix of vectros, i.e. a 3D matrix:
# (a) 5x6 matrix of 2D vectors -- makeVecMatrix( 5, 6, 2 )
# (a) 5x6 matrix of 3D vectors -- makeVecMatrix( 5, 6, 3 )
def makeVecMatrix( rows, cols, vecDim ):
    dtype=numpy.float64
    return numpy.zeros( (rows, cols, vecDim), dtype=dtype )

# returns the number of rows of the given matrix
def getRows(M):
    return M.shape[0]

# returns the number of columns of the given matrix
def getCols(M):
    return M.shape[1]

# returns the dimensions (rows, cols) of the given matrix
def getSize(M):
    return getRows(M), getCols(M)

# saves the given 2D array to the file with the given name
# in binary Numpy format (.npy extension is added)
def saveMatrix(M, filename):
    numpy.save(filename, M)

# loads the given 2D array from the file with
# the given name in binary Numpy format
def loadMatrix(filename):
    return numpy.load(filename)
    


# creates a vector from:
# (a) python list -- makeVector( [1, 2] )
# (b) given (x, y) coords -- makeVector( 1, 2 )
def makeVector( *args ):
    if len(args) != 1:
        # from given coordinates: makeVector(1, 2) or makeVector( 1, 2, 3 )
        return makeMatrix( args )
    else:
        # from list or vector: makeVector( [1, 2] ), makeVector( z )
        coords = [ c for c in args[0] ]
        return makeMatrix( coords )


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

# returns a numpy 2D array of (R, G, B) vectors
def loadImageRGB(filename):
    image = Image.open(filename)
    pixMap = image.load()
    cols, rows = image.size[0], image.size[1]   # width(0)=>cols,height(1)=>rows
    pixels = makeVecMatrix(rows, cols, 3)
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
        # if PIL image
        image.save(filename, "png")
    except:
        # if Numpy matrix
        if len(image.shape) == 3:      # RGB image
            image = mat2rgbimg(image)
        else:                          # GRAY image
            # remap to 0..255
            """
            image = image - numpy.min(image)
            if numpy.max(image) != 0:
                image = image/numpy.max(image)
                image = 255*image
            """
            #image = image*255
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

# creates a black RGB image with the given dimensions
def makeImageRGB(rows, cols):
    image = Image.new("RGB", (cols, rows), (0,0,0))        # cols=>width,rows=>height
    pixels = image.load()
    return image

# converts the given matrix in RGB format, i.e. each pixel (R,G,B)
# to GRAY fromat, i.e. each pixel a single value
def rgb2gray( rgb ):
    rows, cols = getSize( rgb )
    gray = makeMatrix( rows, cols )
    transform = (0.299, 0.587, 0.114)
    for r in range(0, rows):
        for c in range(0, cols):
            gray[r, c] = rgb[r, c].dot(transform)
    return gray

# converts the given matrix to PIL image
def mat2img( M ):
    rows, cols = getSize( M )
    image = makeImage( rows, cols, 0 )
    pixels = image.load()
    for r in range(0, rows):
        for c in range(0, cols):
            pixels[c, r] = int(M[r, c])              # (r,c)=(y,x)=>(x, y)
    return image

# converts the given matrix to RGB PIL image
def mat2rgbimg( M ):
    rows, cols = getSize( M )
    image = makeImageRGB( rows, cols )
    pixels = image.load()
    for r in range(0, rows):
        for c in range(0, cols):
            pixels[c, r] = tuple(numpy.rint(M[r, c]).astype(int))              # (r,c)=(y,x)=>(x, y)
    return image

# converts the given PIL image to matrix
def img2mat( image ):
    width, height = image.size
    rows, cols = height, width
    M = makeMatrix( rows, cols )
    pixels = image.load()
    for r in range(0, rows):
        for c in range(0, cols):
            M[r, c] = pixels[c, r]             # (r,c)=(y,x)=>(x, y)
    return M

# shows a window with the image (either a matrix or PIL image)
def showImage( image ):
    try:
        # if PIL image
        image.show()
    except:
        # if Numpy matrix
        if len(image.shape) == 3:      # RGB image
            image = mat2rgbimg(image)
        else:                          # GRAY image
            # remap to 0..255
            """
            image = image - numpy.min(image)
            if numpy.max(image) != 0:
                image = image/numpy.max(image)
                image = 255*image
            """
            #image = image*255
            image = mat2img( image )
        image.show()
