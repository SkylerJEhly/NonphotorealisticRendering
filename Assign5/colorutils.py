import support
import sys

#Increase recursion depth
sys.setrecursionlimit(100000) 

# colorutils.py

# Assigns solid colors to each of the regions in the drawings
# Goes through each pixel in the drawing and if it is -1 computes the average
#  color of the region for that pixel and assigns that color to each pixel
#  in the region
def assignColors(image, drawing):
    #print "assign"
    rows,cols = support.getSize(drawing)
    rgbImage = support.makeVecMatrix(rows,cols,3)
    for r in range (0,rows,1):
        for c in range(0,cols,1):
            if drawing[r,c] == -1:
                total,numPixels = computeColor(r,c,image,drawing)
                rgbColor = total/numPixels
                #print(rgbColor * 255)
                floodRegion(r,c,rgbColor,drawing, rgbImage)
    return rgbImage * 255
                


# Recursive method that computes the total color and total number of pixels in
#  the region that contains pixel (r,c).
# Returns (0,0) for pixels whose value in the drawing is not -1;
# Otherwise, marks the pixel with -2 and recursively looks to each
#  of the 4 neighbors N,S,E,W
# Returns total, numPixels
def computeColor(r, c, image, drawing):
    #print"compute color"
    rows,cols = support.getSize(drawing)
    
    # base case:
    if isOutOfBounds(rows,cols,r,c) or drawing[r,c] != -1:
        return 0, 0

    # otherwise
    drawing[r,c] = -2
    n_tot, n_pix = computeColor(r-1,c,image,drawing)
    s_tot, s_pix = computeColor(r+1,c,image,drawing)
    e_tot, e_pix  = computeColor(r,c-1,image,drawing)
    w_tot, w_pix  = computeColor(r,c+1,image,drawing)
    total = image[r,c]+n_tot+s_tot+e_tot+w_tot
    numPixels = 1+n_pix+s_pix+e_pix+w_pix
    return total, numPixels


# Recursive method that assigns the given color to each pixel in the region
#  that contains pixel (r,c).
# Does nothing for pixels whose values are not -2;
# Otherwise recursively looks to each of the 4 neighbors N,S,E,W
def floodRegion(r, c, rgbColor, drawing, rgbImage):
    rows,cols = support.getSize(drawing)

    #base case
    # if draw_value != -2
    # do nothing
    
    if not isOutOfBounds(rows,cols,r,c) and drawing[r,c] == -2:
        drawing[r,c] = -3
        rgbImage[r,c] = rgbColor
        #print "flood color", rgbColor
        floodRegion(r-1,c,rgbColor,drawing,rgbImage) #north
        floodRegion(r+1,c,rgbColor,drawing,rgbImage) #south
        floodRegion(r,c-1,rgbColor,drawing,rgbImage) #east
        floodRegion(r,c+1,rgbColor,drawing,rgbImage) #west


def isOutOfBounds(rows,cols, r,c):
    if r >= rows:
        return True
    if r < 0:
        return True
    if c >= cols:
        return True
    if c < 0:
        return True
    return False
