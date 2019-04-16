#linedrawing.py
# Author: Skyler Ehly
# Date: 2/15/2018

# ---------> y
# |
# |
# |
# v
# x

import support
import numpy as np
import math
import utils

#PARAMATERS#
radius = 2        # radius
N = 1             # fall-off rate
sig_c = 1.0       # variance for surrounding sections
sig_s = sig_c*1.6 # variance for the center, determined by sig_c
sig_m = 3.0       # controls the length of the elongated flow kernel 
p_noise = 0.99    # controls the level of noise detected
s_factor = 1      # smoothing factor, number of times to loop to get T'x, T'y
thresh = 0.8      # threshold
delta_m = 1       # step size m
delta_n = 1       # step size n

# Takes an image and produces a line drawing image of the original
def lineDrawing(image):
    # load image and create slightly smaller image due to gradients being smaller
    image_mat = support.loadImage(image)
    r, c = support.getSize(image_mat)
    image_mat = image_mat[1:r-1, 1:c-1]
    #print(image_mat[0:5,0:5])
    #image_mat = image_mat / 255
    #print(image_mat[0:5,0:5])
    
    Kx, Ky = utils.makeKernel('sobel')
    Kx, Ky = normalizeMatrix(Kx), normalizeMatrix(Ky)
    Gx, Gy, G, D = utils.getEdges(image, Kx, Ky)

    Tx, Ty = gradientTangents(Gx, Gy)
    print(Tx[0:10,0:10])
    print(Ty[0:10,0:10])
    Tx_p, Ty_p = computeETF(Tx, Ty, G)
    print(Tx_p[0:10,0:10])
    print(Ty_p[0:10,0:10])
    print("Saving...")
    support.saveImage(Tx, "Tx_p_test")
    support.saveImage(Ty, "Ty__p_test")
    print("Saved")

    return computeFDoG(image_mat, Tx_p, Tx_p)


# Computes the tangent vectors at each of the given gradients
# Returns Tx, Ty
def gradientTangents(Gx, Gy):
    rows, cols = support.getSize(Gx)
    zero_matrix = support.makeMatrix(rows,cols)
    Tx, Ty = zero_matrix, zero_matrix
    for r in range(0, rows, 1):
        for c in range(0, cols, 1):
            vec = support.makeVector(Gx[r,c],Gy[r,c])
            tangent = rotateCCW(vec)
            Tx[r,c], Ty[r,c] = tangent[0], tangent[1]

    return Tx, Ty
            

# Computes the edge tangent flow given in Eq 1
# Returns, T'x, T'y
def computeETF(Tx, Ty, G):

    #EQUATIONS#

    #Spatial Weight Function
    # Eq 2
    # Return 1 if ||x-y|| < r
    # 0 otherwise
    def sWeight(x,y):
        dist = distBetweenVectors(x,y)
        if dist < radius:
            return 1
        return 0
        

    # Magnitute Weight Function
    # Eq 3
    # Returns (1/2)(1+tanh[(g^(y)-g^(x))])
    # g^(z) - normalized gradient magnitude at z
    def mWeight(x,y,G):
        g_hat = normalizeMatrix(G)
        gx_hat = g_hat[x[0], x[1]]
        gy_hat = g_hat[y[0],y[1]]
        return .5*(1+math.tanh(N*(gy_hat-gx_hat)))
        
    # Direction Weight Function
    # Eq 4
    # Returns |tcur(x) (dot) tcur(y)|
    # tcur(z) = current normalized tangent vector at z
    def dWeight(x,y,tcur):
        tcur_x = tcur[x[0],x[1]]
        tcur_y = tcur[y[0],y[1]]
        result = getDotProduct(tcur_x, tcur_y)
        # absolute value of result (always 0 or positive)
        if (result < 0):
            result = -result
        return result

    # Phi
    # Eq 5
    # Return 1 if tcur(x) (dot) tcur(y) > 0
    # Return -1 otherwise
    def phi(x,y,tcur):
        tcur_x = tcur[x[0],x[1]]
        tcur_y = tcur[y[0],y[1]]
        result = getDotProduct(tcur_x, tcur_y)
        if (result < 0):
            return 1
        return -1

    
    rows, cols = support.getSize(Tx)
    Tx_cur, Ty_cur = Tx, Ty
    zero_matrix = support.makeMatrix(rows,cols)
    Tx_new, Ty_new = zero_matrix, zero_matrix

    for i in range(0, s_factor, 1):
        print("iteration: %d" % i) 
        # For every pixel in T
        for r in range(0, rows, 1):
            for c in range(0, cols, 1):
                x = support.makeVector(r,c)
                # For every neighbor of x
                sum_x = 0
                sum_y = 0
                
                # Calculate mask for neighbors of x
                rn_init = r - radius
                rn_end  = r + radius
                cn_init = c - radius
                cn_end  = c + radius
                if rn_init < 0:
                    rn_init = 0
                if rn_end > rows:
                    rn_end = rows
                if cn_init < 0:
                    cn_init = 0
                if cn_end > cols:
                    cn_end = cols

                for rn in range(rn_init, rn_end, 1):
                    for cn in range(cn_init, cn_end, 1):
                        y = support.makeVector(rn,cn)
                        if sWeight(x,y) == 1:
                            mW = mWeight(x,y,G)
                            sum_x = sum_x + phi(x,y,Tx_cur)*Tx_cur[y[0],y[1]]*mW*dWeight(x,y,Tx_cur)
                            sum_y = sum_y + phi(x,y,Ty_cur)*Ty_cur[y[0],y[1]]*mW*dWeight(x,y,Ty_cur)

                #Normalize Vector
                v = support.makeVector(sum_x, sum_y)
                normal_v = normalizeVector(v)          
                Tx_new[r,c] = normal_v[0]
                Ty_new[r,c] = normal_v[1]
                
        Tx_cur = Tx_new
        Ty_cur = Ty_new

    return Tx_cur, Ty_cur



# Computes the flow-based difference-of-gaussians given Eq 10.
# Returns final drawing
def computeFDoG(image, Tx, Ty):

    #VARIABLES#
    p = 6 * sig_m
    q = 6 * sig_c

    #EQUATIONS#

    # Gsig(x)
    # Eq 8
    def gaussian(t, sig):
        return 1/(math.sqrt(2*math.pi)*sig) * math.exp(-(t*t)/(2*sig*sig))

    # f(t)
    # Eq 7
    def filter1d(t):
        return gaussian(t, sig_c) - p_noise * gaussian(t, sig_s)

    # F(s)
    # x is a vector
    # Eq 6
    def filteringFramework(x, Tx, Ty):
        r = x[0]  # row of x
        c = x[1]  # col of x
        end = int(q/2)
            
        total_sum = 0
    
        # Calculate Direction
        tan = support.makeVector(Tx[r,c], Ty[r,c])
        direct = rotateCW(tan)
    
        # Loop forwards (t=1,2,3,4)
        init_x = x
        for t in range(1, end,1):
            z = np.add(x, delta_n*direct)
            z = z.astype(int)
            total_sum = total_sum + image[z[0],z[1]] * filter1d(t)
            x = z
            
        # Loop backwards (t=1,2,3,4)
        x = init_x
        for t in range(1, end,1):
            z = np.subtract(x, delta_n*direct)
            z = z.astype(int)
            total_sum = total_sum + image[z[0],z[1]] * filter1d(t)
            x = z
            
        # t = 0
        return total_sum + image[r,c] * filter1d(0)

    # H(x)
    # Eq 9
    def filterAccumulated(x, Tx, Ty):
        r = x[0]  # row of x
        c = x[1]  # col of x
        rows, cols = support.getSize(Tx)
        end = int(p/2)
        
            
        total_sum = 0
        # Loop forwards (s=1,2,3,4)
        for s in range(1, end,1):
            tan = getTangentVector(Tx, Ty, x)
            if isZeroVector(tan):
                break
            print("Not Zero Vector: loop 1")
            z = np.add(x, delta_m*tan)
            z = z.astype(int)
            if isOutOfBounds(rows, cols, z):
                break
            total_sum = total_sum + gaussian(s,sig_m) * filteringFramework(z, Tx, Ty)
            x = z
        

        # Loop backwards
        x = support.makeVector(r,c) # reset x to s=0
        for s in range(1, end,1):
            tan = getTangentVector(Tx, Ty, x)
            if isZeroVector(tan):
                break
            print("Not Zero Vector: loop 2")
            z = np.subtract(x, delta_m*tan)
            z.astype(int)
            if isOutOfBounds(rows, cols, z):
                break
            total_sum = total_sum + gaussian(s,sig_m) * filteringFramework(z, Tx, Ty)
            x = z

        # s = 0
        x = support.makeVector(r,c) # reset to s = 0
        return gaussian(0,sig_m) * filteringFramework(x, Tx, Ty)



    # OTHER FUNCTIONS #

    # Returns the tangent vector at z, or zero vector if out of bounds
    def getTangentVector(Tx, Ty, z):
        rows,cols = support.getSize(Tx)
        T = support.makeVector(Tx[z[0],z[1]], Ty[z[0],z[1]])
        T_tan = rotateCW(T)
        # round answers to integers
        Tx_tan = int(T_tan[0])
        Ty_tan = int(T_tan[1])
        
        return support.makeVector(Tx_tan, Ty_tan)


    rows, cols = support.getSize(image)
    line_image = support.makeMatrix(rows,cols)
    print(support.getSize(line_image))
    for r in range(0,rows,1):
        print("Row", r)
        for c in range(0,cols,1):
            x = support.makeVector(r,c)
            Hx = filterAccumulated(x, Tx, Ty)
            if Hx < 0:
                line_image[r,c] = 0
            else:
                line_image[r,c] = 1 + np.tanh(Hx)

##            if Hx < 0 and 1 + np.tanh(Hx) < thresh:
##                line_image[r,c] = 0
##            else:
##                line_image[r,c] = 1

    return line_image




## Helper Methods ##
# Vector is an array of length 2 with r = a[0] and c = a[1]
def getVectorLength(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1])

# Returns the euclidean distance between vectors v0 and v1
def distBetweenVectors(v0, v1):
    xdis = v1[0] - v0[0]
    ydis = v1[1] - v1[0]
    return math.sqrt( xdis*xdis + ydis*ydis)

# Returns the dot product of v1 (dot) v2
def getDotProduct(v1, v2):
    return np.dot(v1,v2)

# Rotates the vector 90 degrees clockwise
def rotateCW(v):
    r = v[0]
    c = v[1]
    # if r and c are the same sign
    # return +r -c
    if (r > 0 and c > 0) or (r < 0 and c < 0):
        return support.makeVector(v[0], -v[1])
    # else return -r +c
    return support.makeVector(-v[0], v[1])

# Rotates the vector 90 degrees clockwise
def rotateCCW(v):
    r = v[0]
    c = v[1]
    # if r and c are the same sign
    # return -r +c
    if (r > 0 and c > 0) or (r < 0) and (c < 0):
        return support.makeVector(-v[0], v[1])
    # else return +r -c
    return support.makeVector(v[0], -v[1])

# Returns true if v is the zero vector,
# False otherwise
def isZeroVector(v):
    if v[0] == 0 and v[1] == 0:
        return True
    return False

# Returns a normalized vector of v
def normalizeVector(v):
    r = v[0]
    c = v[1]
    length = getVectorLength(v)
    #print("length: %d" % length)
    if length < .000001 and length > -.00001:
        # divide by 0 (ignore)
        #print("Ignored")
        return r,c
    else:
        return support.makeVector(r/length, c/length)

# Returns a normalized matrix of M
def normalizeMatrix(M):
    min_M = np.min(M)
    max_M = np.max(M)
    # subtract matrix by min
    M = M - min_M
    
    # divide matrix my max - min
    return (M*1.0) / (max_M-min_M)

def isOutOfBounds(rows,cols, v):
    r = v[0]
    c = v[1]
    if r >= rows:
        return True
    if r < 0:
        return True
    if c >= cols:
        return True
    if c < 0:
        return True
    return False


