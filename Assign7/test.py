#test.py

import support
import emboss

images = [("fig1a.png", "fig1b.png"),\
          ("fig4b.png", "fig5a.png"),\
          ("fig4b.png", "fig5c.png"),\
          ("fig4b.png", "fig5e.png"),\
          ("fig7a.png", "fig7b.png")]
image_num = 0

def testNormFields():
  image = images[image_num]
  heightField = support.loadImage(image[1])
  heightField = heightField / 255.0
  colorField = support.loadImageRGB(image[0])/255.0
  normalField = emboss.computeNormals(heightField)
  emboss.renderFields(normalField, colorField)

def testDesktop():
  image = support.loadImageRGB("desktop.png")/255.0
  emboss.emboss(image)

def testEagle():
  image = support.loadImageRGB("eagle.png")/255.0
  emboss.emboss(image)

def testFlowers():
  image = support.loadImageRGB("flowers.png")/255.0
  emboss.emboss(image)

def testRiver():
  image = support.loadImageRGB("riverfront.png")/255.0
  emboss.emboss(image)


testDesktop()
##testEagle()
##testFlowers()
##testRiver()
##testNormFields()
