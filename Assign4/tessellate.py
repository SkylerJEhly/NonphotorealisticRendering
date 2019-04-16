#tessellate.py

import support
import numpy as np
import halftone
import math
import sys

#Increase recursion depth
sys.setrecursionlimit(100000)

##PARAMATERS##
N = 144 # Number of points
A = 4.0 #timer
deltaT = .01 #time step
m = 1 #unit mass
s = 1 #curvature magnitude
#minLen = 10
minDist = 5 #Smin
pointsGen = "grid" #halftone, grid, or file
curveF = lambda t: (500-t)**-.8

curveType = 'Lorentz'
ETFr_file = 'lion/lion-ETFr5.npy'
ETFc_file = 'lion/lion-ETFc5.npy'



# Takes a numpy matrix and produces a tessellated image of the original
# The original image is subdivided into smaller regions painted in solid colors
#  1. Computes initial point distributions
#  2. Computes the ETF
#  3. Draws the curves
#  4. Fills regions with solid color
def tessellate_main(image):

    # 1.
    points = generatePoints(image)

    # 2.
    ETFr = support.loadMatrix(ETFr_file)
    ETFc = support.loadMatrix(ETFc_file)
    

    # 3-4
    image_mat = support.loadImageRGB(image)

    # shrink image
    rows,cols = support.getSize(image_mat)
    image_mat = image_mat[1:rows-1,1:cols-1]

    # divide image by 255 so values are 0-1
    image_mat = image_mat / 255
    
    return tessellate(image_mat, points, ETFr, ETFc)
    


# Draws the curves given the initial point distribution
#   (each point has its coordinates and min distance) and the ETF
# Creates an image initialized to -1 that will represent the final drawing
# The image is modified by the functions to follow and -1 means the pixel
#   has not been assigned a color
def tessellate(image, pointsList, ETFr, ETFc):
    rows,cols = support.getSize(image)
    print("SIZE: %d x %d" % (rows,cols))
    drawing = support.makeMatrix(rows,cols,-1)
    if curveType == 'ETF':
        growCurveETF(pointsList, drawing, ETFr, ETFc)
        print("Curve grown!")
    elif curveType == 'Lorentz':
        growCurveLorentz(pointsList,drawing)

    return assignColors(image,drawing)
    


# Generates a list of N points based on the global paramater pointsGen
#   halftone: uses the algorithm from Assignment 1 run for depth equal to log2N
#   grid: generates a grid of size (sqrt(N))^2 points evenly spaced along rows/cols
#   file: loads the points from file "points.txt"; here N is ignored;
#         each line in the file contains two numbers for (row, col) coordinates of a point
# The returned list (pointsGen) contains tuples (r, c, minDist)
#   for the coordinates of each point and the closest distance to an edge
#   of the image or another point
def generatePoints(image):
    if (pointsGen == 'halftone'):
        depth = int(math.log(N,2))
        #print(depth)
        dot_matrix = halftone.halftone(support.loadImageRGB(image), depth, halftone.renderDots)
        rows,cols = support.getSize(dot_matrix)
        dot_matrix = dot_matrix[1:rows-1,1:cols-1]
        rows,cols = support.getSize(dot_matrix)

        # Find points
        pointsList = []
        for r in range(0,rows,1):
            for c in range(0,cols,1):
                if dot_matrix[r,c] == 0:
                    pointsList.append([r,c])

        # Find minimum distance for each point
        # Ceate new points list with minimum distance
        return generatePointTuples(pointsList, pointsList, rows, cols)
            
    elif (pointsGen == 'grid'):
        rows, cols = support.getSize(support.loadImageRGB(image))
        size = math.sqrt(N)
        r_step = int(rows/size)
        r_init = r_step/2
        c_step = int(cols/size)
        c_init = c_step/2

        # Find points
        pointsList = []
        for r in range (r_init, rows-1, r_step):
            for c in range(c_init, cols-1, c_step):
                pointsList.append([r,c])

        # Find minimum distance for each point
        # Ceate new points list with minimum distance
        return generatePointTuples(pointsList, pointsList, rows, cols)

                
    elif (pointsGen == 'file'):
        f = file("points.txt", "w")
        line = f.readline().strip()

        pointsList = []
        while line != "":
            values = line.split(" ")
            pointsList.append([int(values[0]), int(values[1])])

        # Find minimum distance for each point
        # Ceate new points list with minimum distance
        return generatePointTuples(pointsList, pointsList, rows, cols)
                              
    else:
        raise ValueError('Make sure pointsGen has a valid name')


# Grows a curve from the point with the maximum distance using ETF
# That point is removed from the given list
# Along the way the maximum curve distances of each point is updated (if necessary)
# Points whose distance is smaller than parameter minDist are removed

# The curves are drawn in "drawing"
# Short curves, with length smaller than minLen are not drawn
# Store the points in a list and repaint them in white if necessary

#The result is observed by modifiying the given drawing image
def growCurveETF(pointsList, drawing, ETFr, ETFc):
    rows,cols = support.getSize(drawing)
    print rows, cols

    i = 1
    allPoints = pointsList
    while len(pointsList) > 0:
        print("curve: %d"% i)
        i = i+1
        point = findMaxDistPt(pointsList)
        pointsList.remove(point)

        # update minDist for points
        pointsList = generatePointTuples(pointsList, allPoints, rows, cols)

        r_init = point[0]
        c_init = point[1]
        curvePoints = []

        # Draw Curve Positive
        curve = CurveETF(r_init,c_init, ETFr, ETFc, "pos")
        # while still drawing a curve
        isCurveDone = curve.drawCurveETF(curvePoints, drawing)
        while not isCurveDone:
            isCurveDone = curve.drawCurveETF(curvePoints, drawing)

        # Draw Curve Negative
        curve = CurveETF(r_init,c_init, ETFr, ETFc, "neg")
        # while still drawing a curve
        isCurveDone = curve.drawCurveETF(curvePoints, drawing)
        while not isCurveDone:
            isCurveDone = curve.drawCurveETF(curvePoints, drawing)

        """"
        if (len(curvePoints) < minLen):
            print "curve too small"
            # redraw points white
            for point in curvePoints:
                drawing[point[0], point[1]] = -1
        else:
        """
        allPoints.extend(curvePoints)


# Grows a curve from the point with the maximum distance using Lorentz Equation
#  F = qv x B
# That point is removed from the given list
# Along the way the maximum curve distances of each point is updated (if necessary)
# Points whose distance is smaller than parameter minDist are removed

# The curves are drawn in "drawing"
# Short curves, with length smaller than minLen are not drawn
# Store the points in a list and repaint them in white if necessary

#The result is observed by modifiying the given drawing image
def growCurveLorentz(pointsList, drawing):
    rows,cols = support.getSize(drawing)

    i = 1
    allPoints = pointsList
    while len(pointsList) > 0:
        print("curve: %d"% i)
        i = i+1
        point = findMaxDistPt(pointsList)
        pointsList.remove(point)

        # update minDist for points
        pointsList = generatePointTuples(pointsList, allPoints, rows, cols)
        #print(pointsList)

        r_init = point[0]
        c_init = point[1]
        curvePoints = []

        # Draw Curve Positive
        curve = CurveLor(r_init,c_init, rows, cols, "pos")
        # while still drawing a curve
        curve.drawCurveLor(curvePoints, drawing)

        # Draw Curve Negative
        curve = CurveLor(r_init,c_init, rows, cols, "neg")
        # while still drawing a curve
        curve.drawCurveLor(curvePoints, drawing)

        '''
        if (len(curvePoints) < minLen):
            print "curve too small"
            # redraw points white
            for point in curvePoints:
                drawing[point[0], point[1]] = -1
        else:
        '''
        allPoints.extend(curvePoints)


# Assigns solid colors to each of the regions in the drawings
# Goes through each pixel in the drawing and if it is -1 computes the average
#  color of the region for that pixel and assigns that color to each pixel
#  in the region
def assignColors(image, drawing):
    print "assign"
    rows,cols = support.getSize(drawing)
    rgbImage = support.makeVecMatrix(rows,cols,3)
    for r in range (0,rows,1):
        for c in range(0,cols,1):
            if drawing[r,c] == -1:
                total,numPixels = computeColor(r,c,image,drawing)
                rgbColor = total/numPixels
                print(rgbColor * 255)
                floodRegion(r,c,rgbColor,drawing, rgbImage)
    return rgbImage * 255
                


# Recursive method that computes the total color and total number of pixels in
#  the region that contains pixel (r,c).
# Returns (0,0) for pixels whose value in the drawing is not -1;
# Otherwise, marks the pixel with -2 and recursively looks to each
#  of the 4 neighbors N,S,E,W
# Returns total, numPixels
def computeColor(r, c, image, drawing):
    #print"compute color"
    rows,cols = support.getSize(drawing)
    
    # base case:
    if isOutOfBounds(rows,cols,r,c) or drawing[r,c] != -1:
        return 0, 0

    # otherwise
    drawing[r,c] = -2
    n_tot, n_pix = computeColor(r-1,c,image,drawing)
    s_tot, s_pix = computeColor(r+1,c,image,drawing)
    e_tot, e_pix  = computeColor(r,c-1,image,drawing)
    w_tot, w_pix  = computeColor(r,c+1,image,drawing)
    total = image[r,c]+n_tot+s_tot+e_tot+w_tot
    numPixels = 1+n_pix+s_pix+e_pix+w_pix
    return total, numPixels


# Recursive method that assigns the given color to each pixel in the region
#  that contains pixel (r,c).
# Does nothing for pixels whose values are not -2;
# Otherwise recursively looks to each of the 4 neighbors N,S,E,W
def floodRegion(r, c, rgbColor, drawing, rgbImage):
    rows,cols = support.getSize(drawing)

    #base case
    # if draw_value != -2
    # do nothing
    
    if not isOutOfBounds(rows,cols,r,c) and drawing[r,c] == -2:
        drawing[r,c] = -3
        rgbImage[r,c] = rgbColor
        floodRegion(r-1,c,rgbColor,drawing,rgbImage) #north
        floodRegion(r+1,c,rgbColor,drawing,rgbImage) #south
        floodRegion(r,c-1,rgbColor,drawing,rgbImage) #east
        floodRegion(r,c+1,rgbColor,drawing,rgbImage) #west
    

def getVectorCP(rows, cols, r, c):
    C = support.makeVector(int(rows/2), int(cols/2))
    P = support.makeVector(r,c)
    return normalizeVector( P -  C )

# Vector is an array of length 2 with r = a[0] and c = a[1]
def getVectorLength(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1])

# Returns a normalized vector of v
def normalizeVector(v):
    r = v[0]
    c = v[1]
    length = getVectorLength(v)
    if length < .00001 and length > -.00001:
        # divide by 0 (ignore)
        return 0,0
    else:
        return support.makeVector(r/length, c/length)


def findMinDist(allPoints, rows, cols, p1):
    minDist = 0
    r = p1[0]
    c = p1[1]
    # set minimum distance to closest edge
    minD = min([r,c,rows-r,cols-r])

    # for right now check all points
    for p2 in allPoints:
        # check to make sure p1 and p2 are not same point
        if p1[0] != p2[0] and p1[1] != p2[1]:
            dist = pointDist(p1,p2)
            if dist < minD:
                minD = dist
    return minD

def findMaxDistPt(pointsList):
    maxDist = 0
    maxPoint = None
    for point in pointsList:
        dist = point[2]
        if dist > maxDist:
            maxDist = dist
            maxPoint = point
    return maxPoint

def generatePointTuples(pointList, allPoints, rows, cols):
    pointTuples = []
    for point in pointList:
        minD = findMinDist(allPoints, rows, cols, point)
        if minD > minDist:
            pointTuples.append([point[0], point[1], minD])
    return pointTuples


def pointDist(p1, p2):
    x = p2[0]-p1[0]
    y = p2[1]-p1[1]
    return math.sqrt( x*x + y*y )
    
def isOutOfBounds(rows,cols, r,c):
    if r >= rows:
        return True
    if r < 0:
        return True
    if c >= cols:
        return True
    if c < 0:
        return True
    return False


class CurveETF:
    def __init__(self, r, c, ETFr, ETFc, direction):
        self.ETFr = ETFr
        self.ETFc = ETFc
        self.rows, self.cols = support.getSize(ETFr)
        self.F = support.makeVector(ETFr[r,c], ETFc[r,c])
        if self.F[0] == 0 and self.F[1] == 0:
            self.F = getVectorCP(self.rows,self.cols,r,c)
        if direction == 'neg':
            self.F = -self.F
        self.v = support.makeVector(0,0)
        self.x = support.makeVector(r,c)
        self.prev_x = self.x
        self.a = self.F / m

    def advance(self):
        self.v = self.v + self.a*deltaT
        self.prev_x = self.x
        self.x = self.x + self.v*deltaT

    def updateForce(self):
        r, c = int(self.x[0]), int(self.x[1])
        newF = support.makeVector(self.ETFr[r,c], self.ETFc[r,c])
        # if force is now 0,0 continue using old force, otherwise
        if newF[0] != 0 or newF[1] != 0:
            # if obtuse angle, flip the force to make it acute
            if np.dot(self.F, newF) < 0:
                newF = -newF
            self.F = newF
            self.a = self.F / m

    # Returns True if the curve is done drawing; False otherwise
    def drawCurveETF(self, curvePoints, drawing):
        t = 0.0
        while (t < A):
            # If point is out of bounds return True that the curve has ended
            # If drawing pixel is 0 and previous r,c is not the new r,c return True
            r,c = int(self.x[0]), int(self.x[1])
            prev_r, prev_c = int(self.prev_x[0]), int(self.prev_x[1])
            if isOutOfBounds(self.rows,self.cols,r,c) or (drawing[r,c] == 0 and not (prev_r == r and prev_c == c)):
                return True
            
            # Only draw and add points to curvePoints list if it hasn't been visited yet
            if drawing[r,c] != 0:
                drawing[r,c] = 0
                curvePoints.append([r,c])
            self.advance()
            t = t+deltaT
        self.updateForce()
        return False


class CurveLor:
    def __init__(self,r, c, rows, cols, direction):
        self.t = 0
        self.v = getVectorCP(rows, cols, r, c)
        if direction == 'neg':
            self.v = -self.v
        self.x = support.makeVector(r,c)
        self.prev_x = self.x
        self.rows = rows
        self.cols = cols
        self.dir = 1
 

    def advance(self):
        q = s * curveF(self.t)
        qv = q * self.v
        F = support.makeVector(-qv[1], qv[0]) 
        a = F/m
        self.v = self.v + a*deltaT
        self.prev_x = self.x
        self.x = self.x + self.v*deltaT
        self.t = self.t + deltaT

    def drawCurveLor(self, curvePoints, drawing):
        rows, cols = self.rows, self.cols
        i = 0
        while (i < 500):
            # If point is out of bounds return True that the curve has ended
            # If drawing pixel is 0 and previous r,c is not the new r,c return True
            r,c = int(self.x[0]), int(self.x[1])
            prev_r, prev_c = int(self.prev_x[0]), int(self.prev_x[1])
            if isOutOfBounds(rows,cols,r,c) or (drawing[r,c] == 0 and not (prev_r == r and prev_c == c)):
                break
            
            # Only draw and add points to curvePoints list if it hasn't been visited yet
            if drawing[r,c] != 0:
                drawing[r,c] = 0
                curvePoints.append([r,c])
                i = i + 1
                
            self.advance()

            
            
