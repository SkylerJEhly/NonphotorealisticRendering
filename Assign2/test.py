import halftone
import support

#images to png
support.saveImage(support.loadImage('monalisa.pgm'), 'monalisa')
support.saveImage(support.loadImage('einstein.pgm'), 'einstein')
support.saveImage(support.loadImage('table.pgm'), 'table')
support.saveImage(support.loadImage('butterfly.pgm'), 'butterfly')


def testRenderDots():
        result = halftone.halftone("monalisa.pgm", 14, halftone.renderDots)
        support.saveImage(result, "monalisa_dots")

        result = halftone.halftone("einstein.pgm", 14, halftone.renderDots)
        support.saveImage(result, "einstein_dots")

        result = halftone.halftone("table.pgm", 14, halftone.renderDots)
        support.saveImage(result, "table_dots")

        result = halftone.halftone("butterfly.pgm", 14, halftone.renderDots)
        support.saveImage(result, "butterfly_dots")

def testRenderOutline():
        result = halftone.halftone("monalisa.pgm", 12, halftone.renderOutline)
        support.saveImage(result, "monalisa_outl")

        result = halftone.halftone("einstein.pgm", 12, halftone.renderOutline)
        support.saveImage(result, "einstein_outl")

        result = halftone.halftone("table.pgm", 12, halftone.renderOutline)
        support.saveImage(result, "table_outl")

        result = halftone.halftone("butterfly.pgm", 12, halftone.renderOutline)
        support.saveImage(result, "butterfly_outl")

def testRenderTiles():
        result = halftone.halftone("monalisa.pgm", 12, halftone.renderTiles)
        support.saveImage(result, "monalisa_tiles")

        result = halftone.halftone("einstein.pgm", 12, halftone.renderTiles)
        support.saveImage(result, "einstein_tiles")

        result = halftone.halftone("table.pgm", 12, halftone.renderTiles)
        support.saveImage(result, "table_tiles")

        result = halftone.halftone("butterfly.pgm", 14, halftone.renderTiles)
        support.saveImage(result, "butterfly_tiles")

def testRenderTSP():
        result = halftone.halftone("monalisa.pgm", 14, halftone.renderTSP)
        result.save("monalisa_poly.png", "PNG")

        result = halftone.halftone("einstein.pgm", 14, halftone.renderTSP)
        result.save("einstein_poly.png", "PNG")

        result = halftone.halftone("table.pgm", 14, halftone.renderTSP)
        result.save("table_poly.png", "PNG")

        result = halftone.halftone("butterfly.pgm", 14, halftone.renderTSP)
        result.save("butterfly_poly.png", "PNG")
#testRenderDots()
#testRenderOutline()
#testRenderTiles()
#testRenderTSP()
