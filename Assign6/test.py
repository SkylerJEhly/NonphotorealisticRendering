#test.py

import support
import impress

color_plate = '6'

def testInterpolate():
  val = impress.interpolateTest(14.5, 20.2, 91, 210, 162, 95)
  print(val)

def testMultiply(vec, factor):
  return vec * factor

  
def testDesktop():
  image = support.loadImageRGB("originals/desktop.png")
  image = impress.impress(image)
  support.showImage(image)
  support.saveImage(image, 'desktop_color' + color_plate)

def testEagle():
  image = support.loadImageRGB("originals/eagle.png")
  image = impress.impress(image)
  support.showImage(image)
  support.saveImage(image, 'eagle_color' + color_plate)

def testFlowers():
  image = support.loadImageRGB("originals/flowers.png")
  image = impress.impress(image)
  support.showImage(image)
  support.saveImage(image, 'flowers_color' + color_plate)

def testRiver():
  image = support.loadImageRGB("originals/riverfront.png")
  image = impress.impress(image)
  support.showImage(image)
  support.saveImage(image, 'riverfront_color' + color_plate)

def testTable():
  image = support.loadImageRGB("originals/table.png")
  image = impress.impress(image)
  support.showImage(image)
  support.saveImage(image, 'table_color' + color_plate)

def testTiffany():
  image = support.loadImageRGB("originals/tiffany.png")
  image = impress.impress(image)
  support.showImage(image)
  support.saveImage(image, 'tiffany_color' + color_plate)
  

testDesktop()
##testEagle()
##testFlowers()
##testRiver()
##testTable()
##testTiffany()

##print testMultiply(support.makeVector(-1,1), 4)
