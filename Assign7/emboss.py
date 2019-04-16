#emboss.py

import support
import numpy as np
import math
import os
import impress
import utils

# PARAMATERS #
#samples = 30 #should be at least 2*stroke_length
fixedHeight = 1

heightMap_f = "heightmap.png"
opacityMap_f = "opacitymap.png"

resultFile = "desktop_render2.ppm"


# REQUIRED API #

# Input: Numpy color matrix/image, with RGB in [0..1]
# Output: An image that shows an embossing of the original image
def emboss(image):
  rows, cols = support.getSize(image)
  strokes = generateStrokes(image)
  heightField, colorField = computeField(strokes, rows, cols)
  normalField = computeNormals(heightField)
  renderFields(normalField, colorField)
	
# Input: Numpy color matrix/image, with RGB in [0..1]
# Output: A list of strokes, defined by 2 endpoints, color, and width
def generateStrokes(image):

  rows, cols = support.getSize(image)
  points = impress.generatePoints(image)

  strokes = []
  for p in points:
    s = impress.stroke(p, image)
    s.calcEndpoints()
    strokes.append(s)

  return strokes
  
	
# Input: List of strokes, rows and columns of image
# Output: The height field and color field for the strokes
def computeField(strokesList, rows, cols):
  heightMap  = support.loadImage(heightMap_f)/255.0
  opacityMap = support.loadImage(opacityMap_f)/255.0
  mapRows, mapCols = support.getSize(heightMap)
  
  heightField = support.makeMatrix(rows, cols, 0)
  colorField = support.makeVecMatrix(rows, cols, 3)

  # for each stroke
  print "Total = %d" % len(strokesList)
  for index, stroke in enumerate(strokesList):
##    if index%20 == 0:
##      os.system("pkill -f display")
##      support.showImage( heightField )
    if index % 100 == 0:
      print index
    samples = stroke.length
      
    # compute corner point r and coordinate system
    dirL = stroke.end2 - stroke.end1
    dirW = rotateCW(dirL)
    dirL = impress.normalizeVector(dirL)
    dirW = impress.normalizeVector(dirW)
    r = stroke.end1 - dirW*(stroke.width/2)
    
    # step sizes
    stepL = float(stroke.length) / samples
    stepW = float(stroke.width) / samples

    mapStepL = float(mapCols) / samples
    mapStepW = float(mapRows) / samples

    i = 0
    while i < samples:
      j = 0
      while j < samples:
        x = r + dirL*i*stepL + dirW*j*stepW
        if not impress.isOutOfBounds(rows, cols, x):
          f = impress.interpolate(x[0], x[1], heightField)
          map_c = i*mapStepL
          map_r = j*mapStepW
          h = impress.interpolate(map_r, map_c, heightMap)
          t = impress.interpolate(map_r, map_c, opacityMap)
          x = x.astype(int)
          heightField[x[0], x[1]] = f*(1-t) + h*t
          heightField[x[0], x[1]] += fixedHeight
          colorField[x[0], x[1]] = colorField[x[0], x[1]]*(1-t) + stroke.color*t
        j = j + 1
      i = i + 1

  heightField = normalizeMatrix(heightField)

  return heightField, colorField


        

# Input: The heightfield
# Output: Computes and returns a matrix of the unit normals at each pixel
def computeNormals(heightField):
  print "Computing normals..."
  rows, cols = support.getSize(heightField)
  normalField = support.makeVecMatrix(rows, cols, 3)

  # For each pixel that's not a boundary pixel
  for r in range (1, rows-1, 1):
    for c in range(1, cols-1, 1):
      v = support.makeVector(r,c, heightField[r,c])
      # list of neighbor pixels of v (top, left, bottom, right)
      verts = support.makeVector(support.makeVector(r-1,c, heightField[r-1,c]),\
                                 support.makeVector(r,c-1, heightField[r,c-1]),\
                                 support.makeVector(r+1,c, heightField[r+1,c]),\
                                 support.makeVector(r,c+1, heightField[r,c+1]))
      vects = verts - v #vector from v to each vertex

      norms = [cross(vects[0],vects[1]), cross(vects[1],vects[2]),\
               cross(vects[2],vects[3]), cross(vects[3],vects[0])]

      normalField[r,c] = normalizeVector3D(sum(norms))

  return normalField
      
           
# Input: The normal and color fields
# Writes information from the normal and color fields to file display.c 
def renderFields(normalField, colorField):
  rows, cols = support.getSize(normalField)
  
  f = file("drawing.h", "w")
  print >> f, "glBegin( GL_POINTS );"

  for r in range (0, rows, 1):
    for c in range(0, cols, 1):
      norm = normalField[r,c]
      color = colorField[r,c]
      print >> f, "glNormal3f( %f, %f, %f );" % (norm[1], norm[0], norm[2])
      print >> f, "setColor3f( %f, %f, %f );" % (color[0], color[1], color[2])
      print >> f, "glVertex3f( %d, %d, 0 );" % (c, r)

  print >> f, "glEnd();"
  f.close()

  print "Rendering..."

  os.system("sh compile.sh")
  os.system("renderer  -ppm %s -geometry %dx%d" % (resultFile, cols, rows))

	
# Additional Methods #

def cross(u, v):
  x = u[1]*v[2] - v[1]*u[2]
  y = u[2]*v[0] - v[2]*u[0]
  z = u[0]*v[1] - v[0]*u[1]
  return support.makeVector(x,y,z)


# Returns a normalized vector of v
def normalizeVector3D(v):
  length = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
  if length < .00001 and length > -.00001:
    # divide by 0 (ignore)
    return support.makeVector(0,0,0)
  else:
    return v/length

# Returns a normalized matrix of M
def normalizeMatrix(M):
  min_M = np.min(M)
  max_M = np.max(M)
  # subtract matrix by min
  M = M - min_M
  
  # divide matrix my max - min
  return (M*1.0) / (max_M-min_M)


# Rotates the vector 90 degrees clockwise
def rotateCW(v):
    return support.makeVector(v[1], -v[0])	


	  
	  
