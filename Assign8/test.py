#test.py

import support
from inpaint import *

def testfig6b():
  image = support.loadImage("originals/fig6b.png")
  mask = support.loadMatrix("originals/fig6b-mask.npy")
  image = restoreGray(image, mask)

def testfig6bRGB():
  image = support.loadImageRGB("originals/fig6b-rgb.png")
  mask = support.loadMatrix("originals/fig6b-mask.npy")
  image = restoreRGB(image, mask)

def testfig6a():
  image = support.loadImage("originals/fig6a.png")
  mask = support.loadMatrix("originals/fig6a-mask.npy")
  image = restoreGray(image, mask)

def testfig6aRGB():
  image = support.loadImageRGB("originals/fig6a-rgb.png")
  mask = support.loadMatrix("originals/fig6a-mask.npy")
  image = restoreRGB(image, mask)

def testRect():
  image = support.loadImage("originals/rect.png")
  mask = support.loadMatrix("originals/rect-mask.npy")
  image = restoreGray(image, mask)
  support.saveImage(image, "rect-9999.png")
  
def testfig5():
  image = support.loadImage("originals/fig5.png")
  mask = support.loadMatrix("originals/fig5-mask.npy")
  image = restoreGray(image, mask)
  support.saveImage(image, "fig5-9999.png")

def testAnimate():
  image = support.loadImage("originals/animation.png")
  mask = support.loadMatrix("originals/animation-mask.npy")
  image = restoreGray(image, mask)
  support.saveImage(image, "animation-9999.png")

def testAnimateRGB():
  image = support.loadImageRGB("originals/animation-rgb.png")
  mask = support.loadMatrix("originals/animation-mask.npy")
  image = restoreRGB(image, mask)
  support.saveImage(image, "rgb-animation-9999.png")
  
def testImage():
  image = support.loadImage("fig6b-00.png")
  #image = support.loadImage("originals/fig6b.png")
  print(image)


def testRgbToSphere():
  image = support.loadImageRGB("originals/fig6b-rgb.png")
  support.showImage(image)
  r, theta, phi = rgbToSphere(image)
  rgb_image = sphereToRgb(r, theta, phi)
  support.showImage(rgb_image)

  
#testfig6bRGB()
#testfig6aRGB()
#testRect()
#testfig5()
testAnimateRGB()

  
#testImage()
#testRgbToSphere()
  
