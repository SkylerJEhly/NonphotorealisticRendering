#pixelart.py

import support
import utils
import math
import numpy as np

# PARAMATERS #
ep_cluster = 1
ep_palette = 5
perturb = (.10,0,0) # How much to perturb initially the subclusters
Tf = 1 #The final temperature
alpha = 0.7 # lower the temperature by this factor
beta = 1.1 # Variable used in post process
m = 45 # Value that controls the relative weight between color/distance

printProgress = True
scale = 50 # What to scale the output image by in recreation
shouldScale = True



# REQUIRED API #

# Input: Numpy RGB matrix/image, dimensions of desired image, size of palette
# Output: The pixel art image
def pixelart(imageRGB, rowsOut, colsOut, K):
  rowsIn, colsIn = support.getSize(imageRGB)
  N = rowsOut * colsOut # Number of superpixels
  M = rowsIn * colsIn # Number of input pixels

  # Convert the input image to lab space
  imageLAB = support.rgb2lab(imageRGB)

  # Initialize superpixels, palette, and temperature
  superpixels = initSuperPixels(imageLAB, rowsOut, colsOut)
  palette = initPalette(imageLAB)
  T = initT(imageLAB)

  # While (T > Tf)
  i = 0
  while T > Tf:
    if printProgress:
      print "Iteration: %d T: %d" % (i,T)

    # REFINE superpixels with 1 step of modified SLIC
    refineSuperPixels(imageLAB, superpixels)

    # ASSOCIATE superpixels to colors in the palette
    # Update pckps for each cluster
    for cluster in palette:
      cluster.sub1.updatePckps(superpixels, T, palette)
      cluster.sub2.updatePckps(superpixels, T, palette)
    # Update pck for each cluster
    for cluster in palette:
      cluster.sub1.updatePck(superpixels)
      cluster.sub2.updatePck(superpixels)
      

    # REFINE colors in the palette
    totalChange = refinePalette(superpixels, palette)
    if printProgress:
      print "totalChange", totalChange

    # If (palette converged)
    print "Palette size: %d" % len(palette)
    if totalChange < ep_palette:

      # REDUCE temperature T = aT
      T = alpha * T

      # EXPAND palette
      expandPalette(palette, K)

    support.showImage(convertImage(rowsOut, colsOut, superpixels))
    i += 1

  for cluster in palette:
    print cluster.getPckps("0,0")
    print "color", cluster.getColor()

  # Use colors in palette not just average
  for sp in superpixels.itervalues():
     name = "%d,%d" % (sp.outPos[0], sp.outPos[1])
     maxProb = 0.0
     for cluster in palette:
       #print cluster.sub1.pckps[name]
       prob = cluster.getPckps(name)
       if prob > maxProb:
         maxProb = prob
         sp.color = cluster.getColor()
     #print "prob", prob

  return convertImage(rowsOut, colsOut, superpixels)



# OTHER METHODS #

def convertImage(rowsOut, colsOut, superpixels):

  if not shouldScale:
    global scale
    scale = 1
  
  # convert SuperPixels to matrix image
  result = support.makeVecMatrix(rowsOut*scale, colsOut*scale, 3)
  for sp in superpixels.itervalues():
    r,c = int(sp.outPos[0]), int(sp.outPos[1])
    result[r*scale:(r+1)*scale,c*scale:(c+1)*scale] = sp.ms

  # Post-process
  result[:,:,1] = result[:,:,1] * beta
  result[:,:,2] = result[:,:,2] * beta

  # Convert LAB image to RGB
  return support.lab2rgb(result)
  

# Input: Image in LAB space, dimensions of output image
# Output: List of initialized superpixels
# Intialized superpixels are in a regular grid across thge input image
# Each input pixel is assigned to the nearest superpixel in x,y space
def initSuperPixels(imageLAB, rowsOut, colsOut):
  rowsIn, colsIn = support.getSize(imageLAB)
  N = rowsOut * colsOut
  superpixels = {}

  r_step = rowsIn/rowsOut
  r_init = r_step/2
  c_step = colsIn/colsOut
  c_init = c_step/2

  cur_r = r_init
  cur_c = c_init

  # For each superpixel
  # Initialize outPos and inPos
  for out_r in range(0, rowsOut, 1):
    for out_c in range(0, colsOut, 1):
      name = "%d,%d" % (out_r, out_c) # ex. 4,3 for row 4 col 3

      sp = SuperPixel(support.makeVector(out_r,out_c),\
                      support.makeVector(cur_r,cur_c), N)
      superpixels[name] = sp
      
      cur_c += c_step
    cur_r += r_step
    cur_c = c_init

  # Run SLIC
  refineSuperPixels(imageLAB, superpixels)
  return superpixels


# Input: The input image in LAB space
# Output: A list containing a single Cluster
def initPalette(imageLAB):
  # Calculate the average of each channel to get the average color
  L = imageLAB[:,:,0].mean()
  A = imageLAB[:,:,1].mean()
  B = imageLAB[:,:,2].mean()
  color = support.makeVector(L,A,B)
  return [Cluster(SubCluster(color, 0.5), SubCluster(color+perturb, 0.5))]


# Input: The input image in LAB Space
# Output: The starting temperature
# The starting temperature is 1.1 * the critical temperature (twice the variance of pca)
def initT(imageLAB):
  eig1, eig2, eig3 = support.pca(imageLAB)
  return 1.1 * math.sqrt(2 * eig1[0])


# Input: The input image in LAB space, the list of superpixels, the number of pixels of
#        the output image and the input image
# Output: A refined list of superpixels using SLIC algorithm
#         Positions and colors are updated
def refineSuperPixels(imageLAB, superpixels):
  rowsIn, colsIn = support.getSize(imageLAB)
  N = len(superpixels)
  M = rowsIn * colsIn

  # Create dictionary of pixels where value is tuple of superpixel associated with
  # and difference to that superpixel
  pixels = {}

  # Figure out what pixels are associated with what superpixels
  border = int(2 * math.sqrt(float(M)/N))
  for spName, sp in superpixels.iteritems():
    sp.clear()
    out_r, out_c = sp.outPos
    in_r, in_c = sp.inPos
    # For every pixel in border
    for r in range(int(in_r-border),int(in_r+border+1),1):
      for c in range(int(in_c-border),int(in_c+border+1), 1):
        pixName = (r, c)
        # Boundary check
        if not utils.isOutOfBounds(rowsIn, colsIn, pixName):
          diff = sp.calcDiff(pixName, imageLAB, N, M)
          #Check if pixel is already in the dictionary
          if pixName in pixels:
            curDiff = pixels[pixName][1]
            if diff < curDiff:
              pixels[pixName] = (sp, diff)

          else:
            pixels[pixName] = (sp, diff)
          
  # Calculate superpixel averages
  for pos ,value in pixels.iteritems():
    sp, diff = value
    sp.addPixel(imageLAB, pos)


def refinePalette(superpixels, palette):
  totalChange = 0
  for cluster in palette:
    totalChange += cluster.sub1.refine(superpixels)
    totalChange += cluster.sub2.refine(superpixels)

  return totalChange


def expandPalette(palette, K):
  size = len(palette)
  for i in range(0, size, 1):
    cluster = palette[i]
    if cluster.shouldSplit() and len(palette) < K:
      clust1, clust2 = cluster.split()
      palette.remove(cluster)
      palette.append(clust1)
      palette.append(clust2)


# CLASSES #

# Cluster: This class represents a cluster in the palette.
#          A cluster contains two subclusters
class Cluster:
  def __init__(self, sub1, sub2):
    self.sub1 = sub1
    self.sub2 = sub2

  # Return the probability average of the two subcluster colors
  def getColor(self):
    return (self.sub1.pck * self.sub1.ck +\
            self.sub2.pck * self.sub2.ck) / (self.sub1.pck + self.sub2.pck)

  def getPckps(self, name):
    return self.sub1.pckps[name] + self.sub2.pckps[name]

  def shouldSplit(self):
    return utils.getVectorLength(self.sub1.ck-self.sub2.ck) > ep_cluster 

  # Returns 2 new clusters from the old subclusters
  def split(self):
    ck1 = self.sub1.ck
    ck2 = self.sub2.ck
    pck1 = self.sub1.pck / 2.0
    pck2 = self.sub2.pck / 2.0
    return Cluster(SubCluster(ck1, pck1), SubCluster(ck1+perturb, pck1)),\
           Cluster(SubCluster(ck2, pck2), SubCluster(ck2+perturb, pck2))


# Subcluster: This class represents a subcluster in the palette.
#   ck: LAB color components
#   pck: Probability this color/cluster is assigned to any superpixel
#   pckps: Probability this color/cluster is assigned to superpixel ps
class SubCluster:

  def __init__(self, ck, pck):
    self.ck = ck
    print "init cks", ck
    self.pck = pck
    self.pckps = {} #Initially blank

  def updatePckps(self, superpixels, T, palette):
    # e^-((sp.m-sub.ck)^2 /T)
    def calcEquation(sp, cluster, T):
      cDist = utils.getVectorLength(sp.ms - cluster.ck)
      cDist = cDist * cDist
      eEq = float(math.exp(-(cDist/T)))
      return cluster.pck * Eq

    # Recalculate pckps
    self.pckps = {}
    for sp in superpixels.itervalues():
      name = "%d,%d" % (sp.outPos[0], sp.outPos[1]) # ex. 4,3 for row 4 col 3
      topValue = calcEquation(sp, self, T)
      botValue = 0
      for cluster in palette:
        botValue += calcEquation(sp, cluster.sub1, T) +\
                    calcEquation(sp,cluster.sub2, T)
      value = topValue / botValue
      self.pckps[name] = value

  def updatePck(self, superpixels):

    pck = 0
    for sp in superpixels.itervalues():
      name = "%d,%d" % (sp.outPos[0], sp.outPos[1]) # ex. 4,3 for row 4 col 3
      pck += self.pckps[name] * sp.ps
    self.pck = pck


  def refine(self, superpixels):
    totalChange = 0
    topValue = 0
    for sp in superpixels.itervalues():
      name = "%d,%d" % (sp.outPos[0], sp.outPos[1]) # ex. 4,3 for row 4 col 3
      topValue += (sp.ms * self.pckps[name] * sp.ps)

    newCk = topValue / self.pck
    totalChange = utils.getVectorLength(self.ck - newCk)
    self.ck = newCk
    return totalChange
      


# SuperPixel: This class represents a pixel in the final image
#   outPos: Position in the output image
#   inPos: Current position in the input image
#   ms: Average color of associated pixels from input image
#   ps: Importance value P(s); set to 1.0/N for uniform distribution
#   size: Number of associated pixels from input image
class SuperPixel:

  def __init__(self, outPos, inPos, N):
    self.outPos = outPos
    self.inPos = inPos
    self.ms = 0
    self.ps = 1.0/N
    self.size = 0
    self.totalWeight = support.makeVector(0,0,0)
    self.color = support.makeVector(0,0,0)

  def addPixel(self, imageLAB, pos):
    self.totalWeight += imageLAB[int(pos[0]),int(pos[1])]
    self.size += 1
    self.ms = self.totalWeight / self.size

  def calcDiff(self, pixelPos, imageLAB, N, M):
    dc = utils.getVectorLength(self.ms - imageLAB[int(pixelPos[0]),\
                                                  int(pixelPos[1])]) # Color difference
    dp = utils.getVectorLength(pixelPos - self.inPos)# Positional difference
    return dc + m * math.sqrt(N/M) * dp

  def clear(self):
    self.totalWeight = 0
    self.size = 0

    
  
