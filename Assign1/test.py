# test.py

import utils
import support

#images to png
#support.saveImage(support.loadImage('bike.pgm'), 'bike')
#support.saveImage(support.loadImage('peppers.pgm'), 'peppers')
#support.saveImage(support.loadImage('candies.pgm'), 'candies')

# test getEdges(image, kernelx, kernely)
def testGetEdgesBike():
    # prewitt
    Kx, Ky = utils.makeKernel('prewitt')
    Gx, Gy, G, D = utils.getEdges('bike.pgm', Kx, Ky)
    support.saveImage(Gx, 'bike_prewitt_edges_gx')
    support.saveImage(Gy, 'bike_prewitt_edges_gy')
    support.saveImage(G,  'bike_prewitt_edges_g')

    # sobel
    Kx, Ky = utils.makeKernel('sobel')
    Gx, Gy, G, D = utils.getEdges('bike.pgm', Kx, Ky)
    support.saveImage(Gx, 'bike_sobel_edges_gx')
    support.saveImage(Gy, 'bike_sobel_edges_gy')
    support.saveImage(G,  'bike_sobel_edges_g')

def testGetEdgesCandies():
    # prewitt
    Kx, Ky = utils.makeKernel('prewitt')
    Gx, Gy, G, D = utils.getEdges('candies.pgm', Kx, Ky)
    support.saveImage(Gx, 'candies_prewitt_edges_gx')
    support.saveImage(Gy, 'candies_prewitt_edges_gy')
    support.saveImage(G,  'candies_prewitt_edges_g')

    # sobel
    Kx, Ky = utils.makeKernel('sobel')
    Gx, Gy, G, D = utils.getEdges('candies.pgm', Kx, Ky)
    support.saveImage(Gx, 'candies_sobel_edges_gx')
    support.saveImage(Gy, 'candies_sobel_edges_gy')
    support.saveImage(G,  'candies_sobel_edges_g')

def testGetEdgesPeppers():
    # prewitt
    Kx, Ky = utils.makeKernel('prewitt')
    Gx, Gy, G, D = utils.getEdges('peppers.pgm', Kx, Ky)
    support.saveImage(Gx, 'peppers_prewitt_edges_gx')
    support.saveImage(Gy, 'peppers_prewitt_edges_gy')
    support.saveImage(G,  'peppers_prewitt_edges_g')

    # sobel
    Kx, Ky = utils.makeKernel('sobel')
    Gx, Gy, G, D = utils.getEdges('peppers.pgm', Kx, Ky)
    support.saveImage(Gx, 'peppers_sobel_edges_gx')
    support.saveImage(Gy, 'peppers_sobel_edges_gy')
    support.saveImage(G,  'peppers_sobel_edges_g')

# ----------------------------------------------------------
# test smooth average filters
def testSmoothAverageBike():
    # size 3
    K = utils.makeKernel('average', 3)
    image = utils.smoothImage('bike.pgm', K)
    support.saveImage(image, 'bike_average_3_smooth')
    # size 5
    K = utils.makeKernel('average', 5)
    image = utils.smoothImage('bike.pgm', K)
    support.saveImage(image, 'bike_average_5_smooth')
    # size 7
    K = utils.makeKernel('average', 7)
    image = utils.smoothImage('bike.pgm', K)
    support.saveImage(image, 'bike_average_7_smooth')
    # size 9
    K = utils.makeKernel('average', 9)
    image = utils.smoothImage('bike.pgm', K)
    support.saveImage(image, 'bike_average_9_smooth')

def testSmoothAverageCandies():
    # size 3
    K = utils.makeKernel('average', 3)
    image = utils.smoothImage('candies.pgm', K)
    support.saveImage(image, 'candies_average_3_smooth')
    # size 5
    K = utils.makeKernel('average', 5)
    image = utils.smoothImage('candies.pgm', K)
    support.saveImage(image, 'candies_average_5_smooth')
    # size 7
    K = utils.makeKernel('average', 7)
    image = utils.smoothImage('candies.pgm', K)
    support.saveImage(image, 'candies_average_7_smooth')
    # size 9
    K = utils.makeKernel('average', 9)
    image = utils.smoothImage('candies.pgm', K)
    support.saveImage(image, 'candies_average_9_smooth')

def testSmoothAveragePeppers():
    # size 3
    K = utils.makeKernel('average', 3)
    image = utils.smoothImage('peppers.pgm', K)
    support.saveImage(image, 'peppers_average_3_smooth')
    # size 5
    K = utils.makeKernel('average', 5)
    image = utils.smoothImage('peppers.pgm', K)
    support.saveImage(image, 'peppers_average_5_smooth')
    # size 7
    K = utils.makeKernel('average', 7)
    image = utils.smoothImage('peppers.pgm', K)
    support.saveImage(image, 'peppers_average_7_smooth')
    # size 9
    K = utils.makeKernel('average', 9)
    image = utils.smoothImage('peppers.pgm', K)
    support.saveImage(image, 'peppers_average_9_smooth')

#----------------------------------------------------------------
#test smooth gaussian
def testSmoothGaussBike():
    # sig = 0.5 size = 3
    K = utils.makeKernel('gauss', 3, 0.5)
    image = utils.smoothImage('bike.pgm', K)
    support.saveImage(image, 'bike_gauss_.5_smooth')
    # sig = 1 size = 7
    K = utils.makeKernel('gauss', 7, 1)
    image = utils.smoothImage('bike.pgm', K)
    support.saveImage(image, 'bike_gauss_1_smooth')
    # sig = 2 size = 13
    K = utils.makeKernel('gauss', 13, 2 )
    image = utils.smoothImage('bike.pgm', K)
    support.saveImage(image, 'bike_gauss_2_smooth')
    # sig = 3 size = 19
    K = utils.makeKernel('gauss', 19, 3)
    image = utils.smoothImage('bike.pgm', K)
    support.saveImage(image, 'bike_gauss_3_smooth')

def testSmoothGaussCandies():
    # sig = 0.5 size = 3
    K = utils.makeKernel('gauss', 3, 0.5)
    image = utils.smoothImage('candies.pgm', K)
    support.saveImage(image, 'candies_gauss_.5_smooth')
    # sig = 1 size = 7
    K = utils.makeKernel('gauss', 7, 1)
    image = utils.smoothImage('candies.pgm', K)
    support.saveImage(image, 'candies_gauss_1_smooth')
    # sig = 2 size = 13
    K = utils.makeKernel('gauss', 13, 2 )
    image = utils.smoothImage('candies.pgm', K)
    support.saveImage(image, 'candies_gauss_2_smooth')
    # sig = 3 size = 19
    K = utils.makeKernel('gauss', 19, 3)
    image = utils.smoothImage('candies.pgm', K)
    support.saveImage(image, 'candies_gauss_3_smooth')

def testSmoothGaussPeppers():
    # sig = 0.5 size = 3
    K = utils.makeKernel('gauss', 3, 0.5)
    image = utils.smoothImage('peppers.pgm', K)
    support.saveImage(image, 'peppers_gauss_.5_smooth')
    # sig = 1 size = 7
    K = utils.makeKernel('gauss', 7, 1)
    image = utils.smoothImage('peppers.pgm', K)
    support.saveImage(image, 'peppers_gauss_1_smooth')
    # sig = 2 size = 13
    K = utils.makeKernel('gauss', 13, 2 )
    image = utils.smoothImage('peppers.pgm', K)
    support.saveImage(image, 'peppers_gauss_2_smooth')
    # sig = 3 size = 19
    K = utils.makeKernel('gauss', 19, 3)
    image = utils.smoothImage('peppers.pgm', K)
    support.saveImage(image, 'peppers_gauss_3_smooth')

#testGetEdgesBike()
#testGetEdgesCandies()
#testGetEdgesPeppers()

#testSmoothAverageBike()
#testSmoothAverageCandies()
#testSmoothAveragePeppers()

testSmoothGaussBike()
#testSmoothGaussCandies()
#testSmoothGaussPeppers()

