#test.py

import support
import stipple
import colorutils

def testCentroid():
  M = support.makeMatrix(10,10,1)
  P = stipple.computeP(M)
  print "p", P
  Q = stipple.computeQ(P)
  print "q", Q

  r,c = stipple.computeCentroid(P,Q,2,2,5,5)
  print r,c, "Expect 3.5,3.5"
  r,c = stipple.computeCentroid(P,Q,2,2,6,6)
  print r,c, "Expect 4,4"
  r,c = stipple.computeCentroid(P,Q,2,3,5,7)
  print r,c, "Expect 3.5,5"
  
#testCentroid()

def testEagle(): # 500/2000
  image = support.loadImage('eagle.pgm')
  resultPoints, resultCells = stipple.stipple(image)

  image = support.loadImageRGB('eagle.png')
  image = image / 255
  image = colorutils.assignColors(image, resultCells)
  #support.showImage(image)
  support.saveImage(image, "eagle_2000_0.5")

def testFlowers(): # 500/2000
  image = support.loadImage('flowers.pgm')
  resultPoints, resultCells = stipple.stipple(image)

  image = support.loadImageRGB('flowers.png')
  image = image / 255
  image = colorutils.assignColors(image, resultCells)
  #support.showImage(image)
  support.saveImage(image, "flowers_2000_0.65")

def testGoldhill(): #1500/5000
  image = support.loadImage('goldhill.pgm')
  resultPoints, resultCells = stipple.stipple(image)

  image = support.loadImageRGB('goldhill.png')
  image = image / 255
  image = colorutils.assignColors(image, resultCells)
  #support.showImage(image)
  support.saveImage(image, "goldhill_5000_0.65")

def testMonalisa(): #500/2000
  image = support.loadImage('monalisa.pgm')
  resultPoints, resultCells = stipple.stipple(image)

  image = support.loadImageRGB('monalisa.png')
  image = image / 255
  image = colorutils.assignColors(image, resultCells)
  #support.showImage(image)
  support.saveImage(image, "monalisa_2000_0.65")

def testTable(): #1500/5000
  image = support.loadImage('table.pgm')
  resultPoints, resultCells = stipple.stipple(image)

  image = support.loadImageRGB('table.png')
  image = image / 255
  image = colorutils.assignColors(image, resultCells)
  #support.showImage(image)
  support.saveImage(image, "table_1500_0.65")

def testTiffany(): #500/2000
  image = support.loadImage('tiffany.pgm')
  resultPoints, resultCells = stipple.stipple(image)

  image = support.loadImageRGB('tiffany.png')
  image = image / 255
  image = colorutils.assignColors(image, resultCells)
  #support.showImage(image)
  support.saveImage(image, "tiffany_2000_0.65")


# Group 500/2000
##testEagle()
##testFlowers()
##testMonalisa()
##testTiffany()


# Group 1500/5000
##testGoldhill()
testTable()
