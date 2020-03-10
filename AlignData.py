import numpy as np
import sys, math
from sktensor import dtensor
sys.path.insert(1, '../Testing')


def spatial(data):
    print('Starting spatial alignment')
    seq0 = None
    for seq in data:
        index_inner = [0,9,12]
        for frame in range(0,seq.shape[0], 3):
            frameShape = np.zeros([3,15])
            frameShape[0,:] = seq[frame][:]
            frameShape[1,:] = seq[frame+1][:]
            frameShape[2,:] = seq[frame+2][:]
            if (seq0 is None):
                seq0 = frameShape
            else:
                _, _, transform = procrustes(np.transpose(seq0[:,[0,7,9,12]]), np.transpose(frameShape[:,[0,7,9,12]]), False, True)
                Z = np.matmul(transform['scale']*np.transpose(frameShape),transform['rotation'])
                frameShape = np.transpose(Z)
                triangle_static = seq0[:,index_inner]
                triangle_deform = frameShape[:,index_inner]
                _,_, transform2 = procrustes(np.transpose(triangle_static), np.transpose(triangle_deform), False, True)
                frameShape_transformed = np.matmul(transform2['scale']*np.transpose(frameShape),transform2['rotation'])
                frameShape = np.transpose(frameShape_transformed)
                seq[frame][:] = frameShape[0,:]
                seq[frame+1][:] = frameShape[1,:]
                seq[frame+2][:]= frameShape[2,:]
    print('Finished spatial alignment')
    return data

def temporalLazy(data, nrFrames):
    print('Starting lazy temporal alignment')
    for i, seq in enumerate(data):
        if nrFrames * 3 <= seq.shape[0]:
            newFrames = np.zeros([nrFrames*3,15])
            idx = np.round(np.linspace(0,int(seq.shape[0]/3)-1,nrFrames)).astype(int)
            for j, k in enumerate(idx):
                newFrames[j*3] = seq[k*3]
                newFrames[j*3+1] = seq[k*3+1]
                newFrames[j*3+2] = seq[k*3+2]
            data[i] = newFrames
        else:
            sizeMultiplier = round(nrFrames / (seq.shape[0]/3), 3)
            decimal, integer = math.modf(sizeMultiplier)
            decimal = round(decimal, 3)
            tempList = []
            counter = 0.0
            for j in range(0, seq.shape[0], 3):
                counter += decimal
                combinationValue = 1.0
                if counter >= 1.0:
                    Iterations = integer+1
                else:
                    Iterations = integer
                for k in range(0, int(Iterations)):
                    if j < seq.shape[0]-4:
                        tempList.append(combinationValue*seq[j]+(1-combinationValue)*seq[j+3])
                        tempList.append(combinationValue*seq[j+1]+(1-combinationValue)*seq[j+4])
                        tempList.append(combinationValue*seq[j+2]+(1-combinationValue)*seq[j+5])
                    else:
                        distanceToPrevX = seq[j] - seq[j-3]
                        distanceToPrevY = seq[j+1] - seq[j-2]
                        distanceToPrevZ = seq[j+2] - seq[j-1]
                        tempList.append(combinationValue*seq[j]+(1-combinationValue)*(distanceToPrevX+seq[j]))
                        tempList.append(combinationValue*seq[j+1]+(1-combinationValue)*(distanceToPrevY+seq[j+1]))
                        tempList.append(combinationValue*seq[j+2]+(1-combinationValue)*(distanceToPrevZ+seq[j+2]))
                    if counter >= 1:
                        combinationValue -= 1/(integer+1)
                    else:
                        combinationValue -= 1/integer
                counter, _ = math.modf(counter)
                counter = round(counter, 3)
            if len(tempList) < nrFrames * 3:
                tempList.append(seq[seq.shape[0]-3])
                tempList.append(seq[seq.shape[0]-2])
                tempList.append(seq[seq.shape[0]-1])
            data[i] = np.array(tempList)
    print('Finished lazy temporal alignment')
    return data

def createTensor(data, nrPoints, minNrFrames):
    tensor = np.zeros([nrPoints, data.shape[0], minNrFrames])
    for i, seq in enumerate(data):
        W3 = np.zeros([45, minNrFrames])
        for j in range(0, minNrFrames):
            k = 3*j
            shape = seq[k:k+3,:]
            points = np.zeros([nrPoints])
            for l in range(0, 15):
                points[l*3] = shape[0][l]
                points[(l*3)+2] = shape[1][l]
                points[(l*3)+1] = shape[2][l]
            W3[:, j] = points
        tensor[:, i,:] = W3
    tensor = dtensor(tensor)
    return tensor


# From https://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX  - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform