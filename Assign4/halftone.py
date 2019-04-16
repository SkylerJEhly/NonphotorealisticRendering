# haltone.py
# Author: Skyler Ehly
# Date: 2-8-2018

import support
import numpy as np
import os
from PIL import Image, ImageDraw

# This function takes a numpy matrix/image, subdivision depth, and a rendering function
# Returns a numpy matrix or PIL image representing the final result
#	This function coordinates the activities
#   inverts the image
#   calls the subdivision function
#   apllies the given rendering function
def halftone(image, depth, renderFunc):

    #convert image to numpy matrix
    #image_pix = support.loadImage(image)

    #invert the image
    invert_pix = invertImage(image)

    #initialize rectsList (rectangle at 4 corners of image
    rows, cols = support.getSize(invert_pix)
    #print("size", rows, cols)
    rectsList = []
    rectsList.append(Rectangle(Vertex(0,0, "TL"),Vertex(0,cols-1,"TR"),Vertex(rows-1,cols-1,"BR"),Vertex(rows-1,0,"BL")))

    #subdivide rectangles
    rectsList = subdivide(invert_pix, depth, rectsList)
    
    #call the render function
    return renderFunc(rows, cols, rectsList)

# Subdives the given list of rectangles as described in Algorithm 1
# Repeatedly calls itself with the new subdivision up to the given depth
# Initially called with a list that has only one rectangangle representing the whole image
#   Keep in mind that each rectangle in the given list is subdivied either by horizontal edge or by vertical edge
#   Depends on which dimension is bigger
def subdivide(image, depth, rectsList):

    #base case
    if (depth == 0):
        return rectsList

    #2. Identify an edge which divides the input rectangle, along its longer dimension,
    #   into 2 rectangles such that the total mass (sum of pixels) in each side is equal
    subrects = []
    
    for rect in rectsList:
        v0 = rect.v0
        v1 = rect.v1
        v2 = rect.v2
        v3 = rect.v3
        row1 = v0.r
        row2 = v2.r
        col1 = v0.c
        col2 = v2.c
        #print("1",row1, col1, "2",row2,col2)
        # divide vertically
        if ((row2-row1) < (col2-col1)):
            colSplit = findColSplit(image, row1, col1, row2, col2)
                
            #print("colSplit", colSplit)
            v4_1 = Vertex(row1, colSplit, "TR")
            v5_1 = Vertex(row2, colSplit, "BR")
            v4_2 = Vertex(row1, colSplit, "TL")
            v5_2 = Vertex(row2, colSplit, "BL")
                
            subrects.append(Rectangle(v0, v4_1, v5_1, v3))
            subrects.append(Rectangle(v4_2, v1, v2, v5_2))

        # divide horizontally
        else:
            rowSplit = findRowSplit(image, row1, col1, row2, col2)
            v4_1 = Vertex(rowSplit, col2, "BR")
            v5_1 = Vertex(rowSplit, col1, "BL")
            v4_2 = Vertex(rowSplit, col2, "TR")
            v5_2 = Vertex(rowSplit, col1, "TL")

            subrects.append(Rectangle(v0, v1, v4_1, v5_1))
            subrects.append(Rectangle(v5_2, v4_2, v2, v3))


    #4. Repeat step 1 and 2 recursiverly on each side of the sub-rectangles until
    #   a required recursion depth N is reached

    return subdivide(image, depth-1, subrects)


def renderDots(rows, cols, rectsList):
    image = support.makeMatrix(rows, cols, 255)

    for rect in rectsList:
        rect.draw_center(image)
    
    return image


def renderOutline(rows, cols, rectsList):
    image = support.makeMatrix(rows, cols, 255)

    for rect in rectsList:
        rect.draw_outline(image)
    
    return image

def renderTiles(rows, cols, rectsList):
    image = support.makeMatrix(rows, cols, 255)

    for rect in rectsList:
        rect.draw_tiles(image)
    
    return image

def renderTSP(rows, cols, rectsList):
    image = support.makeMatrix(rows, cols, 255)
    
    # get vertex list of rect centers
    rect_centers = []
    for rect in rectsList:
        row,col = rect.get_center()
        rect_centers.append(Vertex(row,col,""))

    dimension = len(rect_centers)

    # create tour.in using rect_centers
    f = file("tour.in", "w")
    print >> f, "NAME tour.in"
    print >> f, "TYPE TSP"
    print >> f, "DIMENSION %d" % dimension
    print >> f, "EDGE_WEIGHT_TYPE : EUC_2D"
    print >> f, "NODE_COORD_TYPE : TWOD_COORDS"
    print >> f, "NODE_COORD_SECTION"

    i = 0
    for v in rect_centers:
        print >> f, "%d %d %d" % (i,v.r,v.c)
        i = i+1
    f.close()

    # run the TSP program
    os.system("tsp_solver -o tour.out tour.in")

    # read the results from tour.out
    f = file("tour.out")
    line = f.readline().strip()
    line = f.readline() # remove first line

    pairs = []
    while line != "":
        values = line.split(" ")
        v0 = rect_centers[int(values[0])]
        pairs.append((v0.c,v0.r))
        line = f.readline()
        
    v1 = rect_centers[int(values[1])]
    pairs.append((v1.c,v1.r))
    f.close()
    # draw a line between all pairs
    return drawPolyLine(rows, cols,pairs)

def renderLoops(rows, cols, rectsList):
    return 0

# Takes in a numpy matrix representing an image
# Returns an inverted numpy matrix with values 0...1
def invertImage(image):

    invt_image = image

    #0->255, 255->0
    invt_image = (invt_image*-1) + 255

    #normalize to 0...1
    invt_image = invt_image / 255
    
    return invt_image


# Draws a horizontal line in a numpy matrix/image connecting two columns along the same row
# Returns new numpy matrix with drawn line
# v0 is always leftmost
def drawHorzLine(image, v0, v1):
    new_image = image;

    #using numpy slice
    new_image[v0.r, v0.c:v1.c] = 0
    
    return new_image


# Draws a vertical line in a numpy matrix/image connecting two rows along the same col
# Returns new numpy matrix with drawn line
# v0 is always topmost
def drawVertLine(image, v0, v1):
    new_image = image;

    #using numpy slice
    new_image[v0.r:v1.r, v0.c] = 0

    return new_image


# Draws a line connecting the pixels in a given list of pairs (r,c)
# Returns a new numpy matrix with drawn lines
def drawPolyLine(rows, cols, pairs):
    size = cols,rows
    im = Image.new("1", size, 1)
    draw = ImageDraw.Draw(im)
    draw.line(pairs)

    return im
		

# Input: Image, two opposite corners of region 
# Computes the row that splits a give rectangular region horizontally in two regions
#that roughly have equal mass
def findRowSplit(image, r1, c1, r2, c2):

    totalArea = np.sum(image[r1:r2, c1:c2])
    i,j = r1, r2
    while i <= j:
        mid = i + (j-i) / 2
        curr_area = np.sum(image[r1:mid, c1:c2])
        if curr_area > totalArea/2 :
            j = mid - 1
        else: # curr_area < totalArea/2
            i = mid + 1

    return mid


# Input: Image, two opposite corners of region 
# Computes the column that splits a give rectangular region vertically in two regions
#that roughly have equal mass
def findColSplit(image, r1, c1, r2, c2):

    totalArea = np.sum(image[r1:r2, c1:c2])
    i,j = c1, c2
    while i <= j:
        mid = i + (j-i) / 2
        curr_area = np.sum(image[r1:r2, c1:mid])
        if curr_area > totalArea/2 :
            j = mid - 1
        else: # curr_area < totalArea/2
            i = mid + 1

    return mid


# A vertex is described by its (r, c) coordinates
# It also has a field that describes if this is L, R, T. B vertex
# r - row #
# c - col #
# o - orientation (L-left, R-right, T-top, B-bottom)
class Vertex:
    def __init__(self, row, col, orient):
        self.r = row
        self.c = col
        self.o = orient


#A rectangle is described by the vertices of its four corners.
#  v0------v1
#   |       |
#  v3------v2
class Rectangle:
    def __init__(self, vert0, vert1, vert2, vert3):
        self.v0 = vert0
        self.v1 = vert1
        self.v2 = vert2
        self.v3 = vert3

    def get_center(self):
        row = (self.v2.r + self.v0.r) / 2
        col = (self.v2.c + self.v0.c) / 2
        return row,col

    def draw_center(self, image):
        row,col = self.get_center()
        image[row][col] = 0 # black pixel


    def draw_outline(self, image):
        v0 = self.v0
        v1 = self.v1
        v2 = self.v2
        drawHorzLine(image, v0,v1)
        drawVertLine(image, v1,v2)

    def draw_tiles(self, image):
        pad = 1
        v0 = self.v0
        v2 = self.v2
        v3 = self.v3
        drawHorzLine(image, v3,Vertex(v2.r,v2.c-pad, "BR"))
        drawVertLine(image, Vertex(v0.r+pad,v0.c,"TL"),v3)
