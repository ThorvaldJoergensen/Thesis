import numpy as np
import sys, math
from sktensor import dtensor
sys.path.insert(1, '../Testing')

# Align the data spatially using the procrustes analysis
def spatial(data):
    print('Starting spatial alignment')
    # Reference sequence
    seq0 = None
    # Iterate over all the sequences
    for seq in data:
        # Chose left, right shoulder and hip for reference points
        index_inner = [0,9,12]
        # Iterate over all frames in the sequence. Steps of 3 because data structured as (x_1,y_1,z_1,x_2,y_2,z_2,...)
        for frame in range(0,seq.shape[0], 3):
            # Save the values for a frame as a single data structure; a shape
            frameShape = np.zeros([3,15])
            frameShape[0,:] = seq[frame][:]
            frameShape[1,:] = seq[frame+1][:]
            frameShape[2,:] = seq[frame+2][:]
            # If no reference shape, chose current shape (i.e. always chosing the first shape as reference)
            if (seq0 is None):
                seq0 = frameShape
            else:
                # Get transformation values to align current shape to the reference shape by looking at the hip, spine, left- & right-shoulder points
                _, _, transform = procrustes(np.transpose(seq0[:,[0,7,9,12]]), np.transpose(frameShape[:,[0,7,9,12]]), True, True)
                # Perform transformation by scaling and rotating on current shape
                Z = np.matmul(transform['scale']*np.transpose(frameShape),transform['rotation'])
                # Transpose the transformation shape and update current shape
                frameShape = np.transpose(Z)
                # Take reference points from current shape and reference shape
                triangle_reference = seq0[:,index_inner]
                triangle_current = frameShape[:,index_inner]
                # Get transformation values to align current shape to the reference shape by looking at reference points
                _,_, transform2 = procrustes(np.transpose(triangle_reference), np.transpose(triangle_current), True, True)
                # Perform transformation by scaling and rotating on current shape
                frameShape_transformed = np.matmul(transform2['scale']*np.transpose(frameShape),transform2['rotation'])
                # Transpose the transformation shape and update current shape
                frameShape = np.transpose(frameShape_transformed)
                # Update values of the input data
                seq[frame][:] = frameShape[0,:]
                seq[frame+1][:] = frameShape[1,:]
                seq[frame+2][:]= frameShape[2,:]
    print('Finished spatial alignment')
    # Return transformed data
    return data

# Perform lazy temporal alignment by streching or shortening sequences to a given amount of frames
def temporalLazy(data, nrFrames):
    print('Starting lazy temporal alignment')
    # Iterate over all sequences in the data
    for i, seq in enumerate(data):
        # Shorten sequences
        if nrFrames * 3 <= seq.shape[0]:
            # Create data model of current sequence with correct length
            newFrames = np.zeros([nrFrames*3,15])
            # Select the given amount of frames evenly spaced across the length of the sequence
            idx = np.round(np.linspace(0,int(seq.shape[0]/3)-1,nrFrames)).astype(int)
            # Iterate over these indices 
            for j, k in enumerate(idx):
                # Save the values for given frames
                newFrames[j*3] = seq[k*3]
                newFrames[j*3+1] = seq[k*3+1]
                newFrames[j*3+2] = seq[k*3+2]
            # Update data entry with new sequence of fixed length
            data[i] = newFrames
        # Strecth sequences
        else:
            # Compute multiplier for how much longer the given amount of frames is compared to the current sequence length
            sizeMultiplier = round(nrFrames / (seq.shape[0]/3), 3)
            # Split into integers and decimals
            decimal, integer = math.modf(sizeMultiplier)
            decimal = round(decimal, 3)
            # Create temporary list
            tempSeq = []
            # Keep track of current multiplication value
            counter = 0.0
            # Iterate over all frames in the sequence. Steps of 3 because data structured as (x_1,y_1,z_1,x_2,y_2,z_2,...)
            for j in range(0, seq.shape[0], 3):
                # Increment multiplication counter
                counter += decimal
                # Keep track of the ratio of how two frames should be combined
                combinationValue = 1.0
                # Save how many intermediate frames should be created between two frames
                if counter >= 1.0:
                    Iterations = integer+1
                else:
                    Iterations = integer
                # Iterate over the amount of frames to be created
                for k in range(0, int(Iterations)):
                    # If within range of the length of original length
                    if j < seq.shape[0]-4:
                        # Create a new frame by making a linear combination of the current frame and the next frame given a weight (combinationValue) for the frames
                        tempSeq.append(combinationValue*seq[j]+(1-combinationValue)*seq[j+3])
                        tempSeq.append(combinationValue*seq[j+1]+(1-combinationValue)*seq[j+4])
                        tempSeq.append(combinationValue*seq[j+2]+(1-combinationValue)*seq[j+5])
                    # If on the last frame
                    else:
                        # Compute the distance to the previous frame
                        distanceToPrevX = seq[j] - seq[j-3]
                        distanceToPrevY = seq[j+1] - seq[j-2]
                        distanceToPrevZ = seq[j+2] - seq[j-1]
                        # Create a new frame by making a linear combination of the current frame and the current frame plus the distance to the previous frame given a weight (combinationValue) for the frames-
                        # I.e. simulate a new frame continuing in the same direction as the original
                        tempSeq.append(combinationValue*seq[j]+(1-combinationValue)*(distanceToPrevX+seq[j]))
                        tempSeq.append(combinationValue*seq[j+1]+(1-combinationValue)*(distanceToPrevY+seq[j+1]))
                        tempSeq.append(combinationValue*seq[j+2]+(1-combinationValue)*(distanceToPrevZ+seq[j+2]))
                    # Compute new combination value based on how many times the frames should be combined
                    if counter >= 1:
                        combinationValue -= 1/(integer+1)
                    else:
                        combinationValue -= 1/integer
                # Update counters 
                counter, _ = math.modf(counter)
                counter = round(counter, 3)
            # If length doesn't match, repeat last frame 
            if len(tempSeq) < nrFrames * 3:
                tempSeq.append(seq[seq.shape[0]-3])
                tempSeq.append(seq[seq.shape[0]-2])
                tempSeq.append(seq[seq.shape[0]-1])
            # Update data entry with new sequence of fixed length
            data[i] = np.array(tempSeq)
    print('Finished lazy temporal alignment')
    # Return aligned data
    return data

# Create a tensor model based on amount of points and frames
def createTensor(data, nrPoints, nrFrames):
    # Create empty tensor of correct size (#points, #sequences, #frames)
    tensor = np.zeros([nrPoints, data.shape[0], nrFrames])
    # Iterate over the sequences
    for i, seq in enumerate(data):
        # Create temporary data structure of current sequence (#points, #frames)
        W = np.zeros([nrPoints, nrFrames])
        # Iterate over the number of frames
        for j in range(0, nrFrames):
            # Compute index value in original data
            k = 3*j
            # Get the shape for a given frame
            shape = seq[k:k+3,:]
            # Create temporary structure for the points of current frame
            points = np.zeros([nrPoints])
            # Iterate over all the points
            for l in range(0, int(nrPoints/3)):
                # Save the points in new order 
                points[l*3] = shape[0][l]
                points[(l*3)+2] = shape[1][l]
                points[(l*3)+1] = shape[2][l]
            # Update current frame in temp sequence
            W[:, j] = points
        # Update tensor entry for current sequence
        tensor[:, i,:] = W
    # Create a tensor model
    tensor = dtensor(tensor)
    # Return tensor
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