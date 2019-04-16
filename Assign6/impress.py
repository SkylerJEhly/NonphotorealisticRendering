#impress.py

import support
import numpy as np
import utils
import random
from PIL import Image, ImageDraw
import math
import linedrawing

#PARAMATERS#
color_plate = 6 # Controls what type of image is outputted
                #NOTE: Still need to change width/length

wobble = True
clipping = color_plate != 2 and color_plate != 6
##clipping = True #Use this one when you want to change the definition of a color plate
fixedOrientation = color_plate == 2 or color_plate == 3
##fixedOrientation = True #Use this one when you want to change the definition of a color plate
useEtf = color_plate == 5 or color_plate == 6
##useEtf = True #Use this one when you want to change the definition of a color plate

useFile = True #True if using Tr,Tc from file
Tr_file = 'etf/desktop-Tr3.npy'
Tc_file = 'etf/desktop-Tc3.npy'
#G_file  = 'etf/desktop-G.npy'


gaus_r = 5         # Gaussian filter radius
r_color = 30    # Range for perturbing the color
r_sf = support.makeVector(.85,1.15)         # Ranage of scaling factor for perturbing the color
r_strokew = support.makeVector(4,4)    # Range for stroke width/radius
r_strokel = support.makeVector(8,20)    # Range for stroke length
r_theta = 30     # Range for perturbing direction angle (in degrees)
antialiased_sf = 4   # scaling factor for generating the antialiased image
threshold = .05

gaus_sig = (gaus_r*2+1)/6    # Gaussian filter sigma


#REQUIRED API#

# Takes a numpy color matrix/image, with R,G,B in [0..1]
# Produces an image that shows impressionist effect of the orignal image
# - Computes Gradients
# - Magnitudes
# - Draws Strokes
# - etc.
def impress(image):

  color_img = np.copy(image) / 255

  # Intesity image derived
  image = support.rgb2gray(image)
##  rows, cols = support.getSize(image)
##  for r in range(0, rows, 1):
##    for c in range(0, cols, 1):
##      image[r,c] = clamp(image[r,c] * random.uniform(r_sf[0],r_sf[1]))


  # 1. Smooth the image with a Gaussian filter
  gauss = utils.makeKernel('gauss', gaus_r, gaus_sig)
  image = utils.smoothImage(image, gauss)


  # 2. Compute gradients using Sobel filter
  Kx, Ky = utils.makeKernel('sobel')
  Gx, Gy, G, D = utils.getEdges(image, Kx, Ky)
  Tr, Tc = updateGradients(Gx, Gy, G)

  # Shrink image so that it matches convolution results
  r,c = support.getSize(image)
  image = image[1:r-1, 1:c-1]

  # 3. Generate a set of points on a grid to serve as the origins of the strokes
  points = generatePoints(r,c)

  # Create blank PIL image
  im = Image.new('RGB', (c*antialiased_sf,r*antialiased_sf))

  # 4. For each origin, draw a stroke:
  i = 0
  print len(points)
  for p in points:
    s = stroke(p, color_img)
    if i % 1000 == 0:
      print(i)
    i = i+1
    s.draw(im, G, Tr, Tc)

  im.thumbnail((c, r), Image.BICUBIC)  
  
  return im


def generatePoints(rows, cols):
  
  step = r_strokew[0]
  init = step/2

  # Find points
  pointsList = []
  r = init
  while r < rows-1:
    c = init
    while c < cols-1:
      pointsList.append([r,c])
      c = c + step
    r = r + step

  random.shuffle(pointsList)
  
  return pointsList


def interpolate(r,c, M):

  rows, cols = support.getSize(M)
  
  int_r, int_c = int(r), int(c) # Rounds down to nearest integer
  if int_r == rows-1 or int_c == cols-1:
    return M[int_r,int_c]
  
  top_l = M[int_r,int_c]
  top_r = M[int_r,int_c+1]
  bot_l = M[int_r+1,int_c]
  bot_r = M[int_r+1,int_c+1]

  u_hor = r - int_r
  v_hor = 1 - u_hor
  u_ver = c - int_c
  v_ver = 1 - u_ver

  top = u_hor*top_r + v_hor*top_l
  bot = u_hor*bot_r + v_hor*bot_l

  return u_ver*bot + v_ver*top



#OTHER METHODS#

def updateGradients(Gr, Gc, G):

  if useFile:
    Tr = support.loadMatrix(Tr_file)
    Tc = support.loadMatrix(Tc_file)

  else:
    
    rows, cols = support.getSize(G)

    # 1. Compute the median of G and the max of G
    medVal_G = np.median(G)
    max_G = np.max(G)

    # 2. For each G[r,c] below a threshold*max
    for r in range(0, rows, 1):
      for c in range(0,cols, 1):
        val = G[r,c]
        if val < threshold*max_G:
          #compute unit vector from center to r,c
          v = normalizeVector(support.makeVector(r,c) - support.makeVector(int(rows/2), int(cols/2)))
          # scale vector by the median
          v = v * medVal_G
          # set Gr, Gc, G at (r,c to the components and magnitude of the vector)
          Gr[r,c] = v[0]
          Gc[r,c] = v[1]
          G[r,c] = math.sqrt(v[0]*v[0] + v[1]*v[1])

          
    Tr, Tc = linedrawing.gradientTangents(Gr, Gc)
    if useEtf:
      Tr, Tc = linedrawing.computeETF(Tr, Tc, G)
    
  return Tr, Tc
        
# Used for getting color and angler perturb value
def getRandPerturb(perturb):
  return (random.randint(0, perturb) - perturb/2) / 255.0

def getNewDir(Tr_val, Tc_val, perturb):
  perturb = math.radians(perturb)
  if Tc_val == 0:
    #print('zero!')
    theta = 45
  else:
    theta = math.atan(Tr_val/Tc_val)

  if wobble:
    theta = theta + perturb
  return support.makeVector(math.sin(theta), math.cos(theta))

def clamp(value):
  if value < 0:
    value = 0
  elif value > 1:
    value = 1
  return value

def isOutOfBounds(rows,cols, v):
  r = v[0]
  c = v[1]
  if r >= rows:
    return True
  if r < 0:
    return True
  if c >= cols:
    return True
  if c < 0:
    return True
  return False

# Returns a normalized vector of v
def normalizeVector(v):
    r = v[0]
    c = v[1]
    length = math.sqrt(v[0]*v[0] + v[1]*v[1])
    if length < .00001 and length > -.00001:
        # divide by 0 (ignore)
        return support.makeVector(0,0)
    else:
        return support.makeVector(r/length, c/length)


def drawStroke(draw, v0, v1, width, color):
  r = width/2
  # Draw straight line
  draw.line([(v0[1],v0[0]),(v1[1],v1[0])], fill=color, width=width)
  # Draw circle at end0
  draw.ellipse([v0[1]-r,v0[0]-r,v0[1]+r,v0[0]+r], fill=color)
  
  # Draw circle at end1
  draw.ellipse([v1[1]-r,v1[0]-r,v1[1]+r,v1[0]+r], fill=color)
  
  


def interpolateTest(r,c, top_l, top_r, bot_l, bot_r):
  int_r, int_c = int(r), int(c) # Rounds down to nearest integer

  u_hor = r - int_r
  v_hor = 1 - u_hor
  u_ver = c - int_c
  v_ver = 1 - u_ver

  top = u_hor*top_r + v_hor*top_l
  bot = u_hor*bot_r + v_hor*bot_l

  return u_ver*bot + v_ver*top



#CLASSES#

class stroke:

  def __init__(self, p, color_img):
    self.p = p
    self.length = random.randint(r_strokel[0], r_strokel[1]+1)
    self.width = antialiased_sf * random.randint(r_strokew[0], r_strokew[1]+1)
    self.thetap = getRandPerturb(r_theta)
    val = interpolate(p[0], p[1], color_img)
    r = clamp(val[0] + getRandPerturb(r_color)) * 255
    g = clamp(val[1] + getRandPerturb(r_color)) * 255
    b = clamp(val[2] + getRandPerturb(r_color)) * 255
    self.color = (int(r), int(g), int(b))

  # Used for color_plate 2 and 3 (fixed orientation)
  def draw(self, drawing, G, Tr, Tc):
    pix_map = drawing.load()
    draw = ImageDraw.Draw(drawing)
    rows, cols = support.getSize(G)
    p = np.around(self.p)

    # Go forward
    x = p
    x_prev = x
    i = -1
    lastSample = 1000
    while not isOutOfBounds(rows, cols, x) and i < self.length/2:
      drawStroke(draw, x_prev*antialiased_sf, x*antialiased_sf, self.width, self.color) 
      
      newSample = interpolate(x[0], x[1], G)
      if (clipping and newSample > lastSample):
        #print "clip"
        break
      
      x_prev = x
      if fixedOrientation:
        x = x + support.makeVector(-1,1) #45 degrees
      else:
        x = x + getNewDir(Tr[x[0], x[1]], Tc[x[0], x[1]], self.thetap)
      lastSample = newSample
      
      i = i+1
    
    # Go backwards
    points = []
    x = p
    x_prev = x
    i = 0
    lastSample = 1000
    while not isOutOfBounds(rows, cols, x) and i < self.length/2:
      if not i == 0:
        drawStroke(draw, x_prev*antialiased_sf, x*antialiased_sf, self.width, self.color) 
      newSample = interpolate(x[0], x[1], G)
      if (clipping and newSample > lastSample):
        #print "clip"
        break

      x_prev = x
      if fixedOrientation:
        x = x + support.makeVector(1,-1) #45 degrees
      else:
        x = x - getNewDir(Tr[x[0], x[1]], Tc[x[0], x[1]], self.thetap)       
      lastSample = newSample
      
      i = i+1
      
