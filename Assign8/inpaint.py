#inpaint.py

import support
import math
from utils import *

# PARAMETERS #
lift = 0.01 # the Lifting Paramater
lamb = 10000 # Lagrange multiplier
T = 9999 # Number of iterations
imgName = "animation" # only used in saving intermediate files
step = 100 # Determines after how many iterations an image is saved


# REQUIRED API#

# Input: A numpy gray matrix/image and a mask indicating the damaged pixels and the regions borders.
# Output: The corrected image
def restoreGray(image, mask):

  # Make a copy of the image
  newImage = image[:]
  support.showImage(newImage)

  # For a certain amount of time run inpaint algorithm
  for i in range(0, T+1, 1):
    inpaint(image, newImage, mask)
    image, newImage = newImage, image
    
    # Every so many iterations save the result
    if i % step == 0:
      support.saveImage(newImage, "%s-%04d.png" % (imgName, i))
      support.showImage(newImage)
      
  return newImage
	

# Input: The current image, the new image, and the mask
def inpaint(curImage, newImage, mask):
  rows, cols = support.getSize(mask)
  # For every vertex
  for i in range(0, rows, 1):
    mask_entry = mask[i,0]
    r = int(mask_entry[0])
    c = int(mask_entry[1])
    t = mask_entry[2]
        
    O = curImage[r,c]
    
    E  = curImage[r,c+1]
    SE = curImage[r+1,c+1]
    S  = curImage[r+1,c]
    SW = curImage[r+1,c-1]
    W  = curImage[r,c-1]
    NW = curImage[r-1,c-1]
    N  = curImage[r-1,c]
    NE = curImage[r-1,c+1]

    # Get magnitudes of midpoints [Eq 6.15]
    mag_e = getMagnitude(O, E, NE, N, S, SE)
    mag_s = getMagnitude(O, S, SE, E, W, SW)
    mag_w = getMagnitude(O, W, SW, S, N, NW)
    mag_n = getMagnitude(O, N, NW, W, E, NE)

    neighbors = [E, S, W, N]
    weights = [wp(mag_e), wp(mag_s), wp(mag_w), wp(mag_n)] #[Eq 6.17/6.22]
    weightsum = sum(weights)
    lagr = lamb * (1 - t)
    
    hoo = lagr / (weightsum + lagr) #[Eq 6.19]

    # Calculate the hop sum [6.18]
    hop = [ weight / (weightsum + lagr) for weight in weights ]

    # Sum of hop times each vertex [Eq 6.20]
    hopup = 0
    for h,v in zip(hop, neighbors):
      hopup += h * v

    newImage[r,c] = hopup + hoo*curImage[r,c] #[Eq 6.21]
	

# Input: A numpy gray matrix/image and a mask indicating the damaged pixels and the regions borders.
# Output: The corrected image
def restoreRGB(image, mask):

  # split the image into spherical coordinates
  r, theta, phi = rgbToSphere(image)
  
  # Make a copy of the matrices
  rNew = r[:]
  tNew = theta[:]
  pNew = phi[:]
  
  # For a certain amount of time run inpaint algorithm
  for i in range(0, T+1, 1):

    # inpaint for each sphere matrix
    inpaint(r, rNew, mask)
    inpaint(theta, tNew, mask)
    inpaint(phi, pNew, mask)

    
    # curImage = newImage, newImage = curImage
    r, rNew     = rNew, r
    theta, tNew = tNew, theta
    phi, pNew   = pNew, phi
    
    # Every so many iterations save the result
    if i % step == 0:
      rgb_image = sphereToRgb(rNew, tNew, pNew)
      support.saveImage(rgb_image, "rgb-%s-%04d.png" % (imgName, i))
      #support.showImage(rgb_image)
  
  return sphereToRgb(rNew, tNew, pNew)
	
	
# OTHER METHODS #

# sqrt( (v1 - O)^2 + [(v2 + v3 - v4 - v5)/4]^2 )
def getMagnitude(O, v1, v2, v3, v4, v5):
  eq1 = v1 - O
  eq2 = (v2 + v3 - v4 - v5)/4.0
  return np.sqrt(eq1*eq1 + eq2*eq2)

#Get pixel weight
def wp(uP):
  return 1.0 / math.sqrt(lift*lift + uP*uP)

def rgbToSphere(rgb_image):
  x_mat = rgb_image[:,:,0]
  y_mat = rgb_image[:,:,1]
  z_mat = rgb_image[:,:,2]
  r = np.sqrt(x_mat*x_mat + y_mat*y_mat + z_mat*z_mat)
  theta = np.arccos(z_mat/r)
  phi = np.arctan2(y_mat,x_mat)
  return r, theta, phi	
	
def sphereToRgb(r_mat, t_mat, p_mat):
  rows, cols = support.getSize(r_mat)
  x = r_mat * np.sin(t_mat) * np.cos(p_mat)
  y = r_mat * np.sin(t_mat) * np.sin(p_mat)
  z = r_mat * np.cos(t_mat)

  # load x, y, z into new matrix
  rgb_image = support.makeVecMatrix(rows, cols, 3)
  rgb_image[:,:,0] = x
  rgb_image[:,:,1] = y
  rgb_image[:,:,2] = z

  return rgb_image



