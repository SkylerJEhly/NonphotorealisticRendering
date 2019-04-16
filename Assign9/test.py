#test.py

import support
from pixelart import *

image = support.loadImageRGB("obama80x54.png")
image = pixelart(image, 6, 4, 4)
support.showImage(image)
support.saveImage(image, "obama6x4")
