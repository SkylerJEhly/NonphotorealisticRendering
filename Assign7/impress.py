#impress.py

import random
import utils
import linedrawing
import math
from PIL import Image, ImageDraw
import numpy as np
import support

#PARAMATERS#
wobble = False
useFile = True #True if using Tr,Tc from file
fixedDir = True # Only calculate one direction at the center of stroke

Tr_file = 'etf/desktop-Tr3.npy'
Tc_file = 'etf/desktop-Tc3.npy'
Tr = None
Tc = None

gaus_r = 5         # Gaussian filter radius
r_color = 30    # Range for perturbing the color
r_strokew = support.makeVector(8,8)    # Range for stroke width/radius
r_strokel = support.makeVector(20,30)    # Range for stroke length
r_theta = 30     # Range for perturbing direction angle (in degrees)
threshold = .05

gaus_sig = (gaus_r*2+1)/6    # Gaussian filter sigma

def generatePoints(image):
  rows, cols = support.getSize(image)

  col_img = np.copy(image)
  image = support.rgb2gray(image*255)

  # 1. Smooth the image with a Gaussian filter
  gauss = utils.makeKernel('gauss', gaus_r, gaus_sig)
  image = utils.smoothImage(image, gauss)

  # 2. Compute gradients using Sobel filter
  Kx, Ky = utils.makeKernel('sobel')
  Gx, Gy, G, D = utils.getEdges(image, Kx, Ky)
  Tr, Tc = updateGradients(Gx, Gy, G)
  
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


def updateGradients(Gr, Gc, G):

  global Tr, Tc

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

def clamp(value):
  if value < 0:
    value = 0
  elif value > 1:
    value = 1
  return value

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

# Vector is an array of length 2 with r = a[0] and c = a[1]
def getVectorLength(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1])

#CLASSES#

class stroke:

  def __init__(self, p, color_img):
    self.p = p
    self.length = random.randint(r_strokel[0], r_strokel[1]+1)
    self.width = random.randint(r_strokew[0], r_strokew[1]+1)
    self.thetap = getRandPerturb(r_theta)
    self.dir = support.makeVector(Tr[p[0],p[1]], Tc[p[0],p[1]])
    self.end1 = (0,0)
    self.end2 = (0,0)
    val = interpolate(p[0], p[1], color_img)
    r = clamp(val[0] + getRandPerturb(r_color))
    g = clamp(val[1] + getRandPerturb(r_color))
    b = clamp(val[2] + getRandPerturb(r_color))
    self.color = support.makeVector(r,g,b) # is [0..1]

  def calcEndpoints(self):
    rows, cols = support.getSize(Tr)
    p = np.around(self.p)

    # Go forward
    x = p
    i = -1
    cur_len = 0
    while not isOutOfBounds(rows, cols, x) and i < self.length/2:
      self.end1 = x
      x = x + self.getNewDir(x, self.thetap)
      i = i+1
      cur_len = cur_len + 1
    
    
    # Go backwards
    x = p
    while not isOutOfBounds(rows, cols, x) and cur_len < self.length:
      self.end2 = x
      x = x - self.getNewDir(x, self.thetap)
      cur_len = cur_len + 1
    
    # If we haven't reached stroke length yet continue in forward direction
    if cur_len != self.length:
      x = self.end1
      while not isOutOfBounds(rows, cols, x) and cur_len < self.length:
        self.end1 = x
        x = x + self.getNewDir(x, self.thetap)
        cur_len = cur_len + 1

    if cur_len != self.length:
      self.length = cur_len

  def getNewDir(self, x, perturb):

    x = x.astype(int)
    Tr_val = Tr[x[0], x[1]]
    Tc_val = Tc[x[0], x[1]]

    if fixedDir:
      return self.dir

    elif wobble:
      theta = math.atan2(Tr_val,Tc_val)
      perturb = math.radians(perturb)
      theta = theta + perturb
      return support.makeVector(math.sin(theta), math.cos(theta))
    
    else:
      return support.makeVector(Tr_val, Tc_val)
