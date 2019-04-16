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
radius = 4        # radius
eta = 1           # fall-off rate
sig_c = 1.0       # variance for surrounding sections
sig_s = sig_c*1.6 # variance for the center, determined by sig_c
sig_m = 3.0       # controls the length of the elongated flow kernel 
p_noise = 1.002    # controls the level of noise detected
s_factor = 1      # smoothing factor, number of times to loop to get T'x, T'y
thresh = 0.8      # threshold
delta_m = 1       # step size m
delta_n = 1       # step size n

# Takes an image and produces a line drawing image of the original
def lineDrawing(image):
    # load image and create slightly smaller image due to gradients being smaller
    image_mat = support.loadImage(image)
    r, c = support.getSize(image_mat)
    image_mat = image_mat / 255.0
    
    Kx, Ky = utils.makeKernel('sobel')
    Kx, Ky = Kx/8.0, Ky/8.0
    Gx, Gy, G, D = utils.getEdges(image, Kx, Ky)
    image_mat = image_mat[1:r-1, 1:c-1]
    G = normalizeMatrix(G)

    Tx, Ty = gradientTangents(Gx, Gy)
    support.saveImage(Tx, "Tx_test")
    support.saveImage(Ty, "Ty_test")
    support.saveImage(Gx, "Gx_test")
    support.saveImage(Gy, "Gy_test")
    Tx_p, Ty_p = computeETF(Tx, Ty, G)
    support.saveImage(Tx, "Tx_p_test")
    support.saveImage(Ty, "Ty_p_test")

    return computeFDoG(image_mat, Tx_p, Tx_p)


# Computes the tangent vectors at each of the given gradients
# Returns Tx, Ty
def gradientTangents(Gx, Gy):
    rows, cols = support.getSize(Gx)
    Tx = support.makeMatrix(rows,cols)
    Ty = support.makeMatrix(rows,cols)
    for r in range(0, rows, 1):
        for c in range(0, cols, 1):
            vec = support.makeVector(Gx[r,c],Gy[r,c])
            tangent = rotateCCW(vec)
            normaltangent = normalizeVector(tangent)
            Tx[r,c], Ty[r,c] = normaltangent[0], normaltangent[1]
                        
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
        dist = getVectorLength(x - y)
        if dist < radius:
            return 1
        return 0
        

    # Magnitute Weight Function
    # Eq 3
    # Returns (1/2)(1+tanh[(g^(y)-g^(x))])
    # g^(z) - normalized gradient magnitude at z
    def mWeight(x,y):
        gx_hat = G[x[0],x[1]]
        gy_hat = G[y[0],y[1]]
        return .5*(1+math.tanh(eta*(gy_hat-gx_hat)))
        
    # Direction Weight Function
    # Eq 4
    # Returns |tcur(x) (dot) tcur(y)|
    # tcur(z) = current normalized tangent vector at z
    def dWeight(x,y):
        T_at_x = support.makeVector(Tx[x[0],x[1]], Ty[x[0],x[1]])
        T_at_y = support.makeVector(Tx[y[0],y[1]], Ty[y[0],y[1]])
        result = getDotProduct(T_at_x, T_at_y)
        # absolute value of result (always 0 or positive)
        if (result < 0):
            result = -result
        #print(tcur_x, tcur_y)
        #print(result)
        return result

    # Phi
    # Eq 5
    # Return 1 if tcur(x) (dot) tcur(y) > 0
    # Return -1 otherwise
    def phi(x,y):
        T_at_x = support.makeVector(Tx[x[0],x[1]], Ty[x[0],x[1]])
        T_at_y = support.makeVector(Tx[y[0],y[1]], Ty[y[0],y[1]])
        result = getDotProduct(T_at_x, T_at_y)
        if (result > 0):
            return 1
        return -1

    
    rows, cols = support.getSize(Tx)
    Tx_new = support.makeMatrix(rows,cols)
    Ty_new = support.makeMatrix(rows,cols)
    print "ETF start"
    for i in range(0, s_factor, 1):
        print("iteration: %d" % i) 
        # For every pixel in T
        for r in range(0, rows, 1):
            print "Row", r
            for c in range(0, cols, 1):
                x = support.makeVector(r,c)
                sum_x = 0
                sum_y = 0
                
                # Calculate mask for neighbors of x
                rn_init = r - radius
                rn_end  = r + radius
                cn_init = c - radius
                cn_end  = c + radius
                    
                # For each neighbor within radius
                for rn in range(rn_init, rn_end+1, 1):
                    for cn in range(cn_init, cn_end+1, 1):
                        y = support.makeVector(rn,cn)
                        if isOutOfBounds(rows,cols,y):
                            break
                        if sWeight(x,y) == 1:
                            mW = mWeight(x,y)
                            dW = dWeight(x,y)
                            phi_eq = phi(x,y)
                            #print(x, y)
                            sum_x = sum_x + phi_eq*Tx[y[0],y[1]]*mW*dW
                            sum_y = sum_y + phi_eq*Ty[y[0],y[1]]*mW*dW

                #Normalize Vector
                v = support.makeVector(sum_x, sum_y)
                normal_v = normalizeVector(v)
                Tx_new[r,c] = normal_v[0]
                Ty_new[r,c] = normal_v[1]
                
        Tx = Tx_new
        Ty = Ty_new

    return Tx, Ty



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
        return 1.0/(math.sqrt(2*math.pi)*sig) * math.exp(-(t*t)/(2*sig*sig))

    # f(t)
    # Eq 7
    def filter1d(t):
        return gaussian(t, sig_c) - p_noise * gaussian(t, sig_s)

    # F(s)
    # x is a vector
    # Eq 6
    def filteringFramework(x):
        r = int(x[0])  # row of x
        c = int(x[1])  # col of x
        end = int(p/2)

        # t = 0
        total_sum = image[r,c] * filter1d(0)
    
        # Calculate Direction
        tan = support.makeVector(Tx[r,c], Ty[r,c])
        direct = rotateCW(tan)
    
        # Loop forwards (t=1,2,3,4)
        z = x
        #print "t forward"
        for t in range(1, end+1,1):
            z = z + delta_n*direct
            z_int = z.astype(int)
            #print(z_int)
            if isOutOfBounds(rows, cols, z_int):
                #print "out of bounds 1", z_int, t
                break
            total_sum = total_sum + image[z_int[0],z_int[1]] * filter1d(t)
            
        # Loop backwards (t=1,2,3,4)
        z = x
        #print "t backward"
        for t in range(1, end+1,1):
            z = x - delta_n*direct
            z_int = z.astype(int)
            #print(z_int)
            if isOutOfBounds(rows, cols, z_int):
                #print "out of bounds 2", z_int, t
                break
            total_sum = total_sum + image[z_int[0],z_int[1]] * filter1d(t)

        return total_sum

    # H(x)
    # Eq 9
    def filterAccumulated(x):
        rows, cols = support.getSize(Tx)
        end = int(q/2)
        
        # s = 0  
        total_sum = gaussian(0,sig_m) * filteringFramework(x)

        z = x        
        # Loop forwards (s=1,2,3,4)
        #print "s forward"
        for s in range(1, end+1,1):
            tan = getTangentVector(z)
            if isZeroVector(tan):
                break
            z = z + delta_m*tan
            z_int = z.astype(int)
            #print(z_int)
            if isOutOfBounds(rows, cols, z_int):
                #print "out of bounds 3", z_int, s
                break
            total_sum = total_sum + gaussian(s,sig_m) * filteringFramework(z)
            

        # Loop backwards
        z = x
        #print "s backward"
        for s in range(1, end+1,1):
            tan = getTangentVector(z)
            if isZeroVector(tan):
                break
            z = z - delta_m*tan
            z_int = z.astype(int)
            #print(z_int)
            if isOutOfBounds(rows, cols, z_int):
                #print "out of bounds 4", z_int, s
                break
            total_sum = total_sum + gaussian(s,sig_m) * filteringFramework(z)

        return total_sum


    # OTHER FUNCTIONS #

    # Returns the tangent vector at z, or zero vector if out of bounds
    def getTangentVector(z):
        z = z.astype(int)
        return support.makeVector(Tx[z[0],z[1]], Ty[z[0],z[1]])

    


    rows, cols = support.getSize(image)
    line_image = support.makeMatrix(rows,cols)
    print(support.getSize(line_image))
    for r in range(0,rows,1):
        print "Row", r
        for c in range(0,cols,1):
            x = support.makeVector(r,c)
            Hx = filterAccumulated(x)
            #print (Hx)
            if Hx > 0:
                line_image[r,c] = 1
            else:
                line_image[r,c] = 1 + np.tanh(Hx)

##            if Hx < 0 and 1 + np.tanh(Hx) < thresh:
##                line_image[r,c] =0
##            else:
##                line_image[r,c] = 1

    return line_image




## Helper Methods ##
# Vector is an array of length 2 with r = a[0] and c = a[1]
def getVectorLength(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1])

# Returns the dot product of v1 (dot) v2
def getDotProduct(v1, v2):
    return np.dot(v1,v2)

# Rotates the vector 90 degrees clockwise
def rotateCW(v):
    return support.makeVector(v[1], -v[0])

# Rotates the vector 90 degrees clockwise
def rotateCCW(v):
    return support.makeVector(-v[1], v[0])
    
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
    if length < .00001 and length > -.00001:
        # divide by 0 (ignore)
        return support.makeVector(0,0)
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


