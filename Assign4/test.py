import tessellate
import support

def testEagle(name):
    image = tessellate.tessellate_main('eagle/eagle.png')
    support.saveImage(image, name)

def testLotus(name):
    image = tessellate.tessellate_main('lotus/lotus.png')
    support.saveImage(image, name)

def testLion(name):
    image = tessellate.tessellate_main('lion/lion.png')
    support.saveImage(image, name)

def testTiffany(name):
    image = tessellate.tessellate_main('tiffany/tiffany.png')
    support.saveImage(image, name)


# test rects-in
def testRects():
    image = support.loadImageRGB("rects-in.png")
    drawing = support.loadMatrix("rects-curves.npy")
    image = image/255.0
    rgbImage = tessellate.assignColors(image,drawing)
    support.saveImage(rgbImage, "rects-test.png")


# test lotus-in
def testLotusIn():
    image = support.loadImageRGB("lotus-in.png")
    drawing = support.loadMatrix("lotus-curves.npy")
    image = image/255.0
    rgbImage = tessellate.assignColors(image,drawing)
    support.saveImage(rgbImage, "lotus-test.png")


#testRects()
#testEagle('eagle_grid_lor_16')
#testLotus('lotus_halftone_lor_10')
testLion('lion_grid_lor_64')
#testTiffany('tiffany_grid_lor_144')
