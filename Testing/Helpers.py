import numpy as np
from sktensor import dtensor
from sktensor.tucker import hosvd
import glob, os, math
from scipy.io.matlab import loadmat


# Perform HOSVD and return core tensor and U matrices
def svd(tensor):
    print('Starting HOSVD')
    # Perform HOSVD with given dimensions
    U, core_S = hosvd(tensor, [tensor.shape[0], tensor.shape[1], tensor.shape[2]])

    # Core model
    core_S = np.array(core_S)

    # U vectors
    U1 = U[0]
    U2 = U[1]
    U3 = U[2]

    print('Fnished HOSVD')
    return U1, U2, U3, core_S

# Create a tensor model from a core tensor and the three U matrices
def createTensor(core_S, U1, U2, U3):
    firstProduct = np.tensordot(core_S, U1, (0,1))
    secodnProduct = np.tensordot(firstProduct,U2, (0,1))
    finalProduct = np.tensordot(secodnProduct,U3,(0,1))

    tensor = dtensor(finalProduct)
    return tensor
# Compute mean shape tensor from given tensor subselection
def createMean(tensor):
    mean_new_shape = np.mean(tensor, axis=(2,1))
    mean_new_shape.reshape(45,1)
    mean_tensor = np.zeros((45,tensor.shape[1],128))
    for i, x in enumerate(mean_new_shape):
        mean_tensor[i] = np.resize(x, (tensor.shape[1],128))
    return mean_tensor

# Selects a given predefined subset of sequences
def getSequence(tensor, labels, name = 'all'):
    allJump = tensor[:,18:35,:]
    allRun = tensor[:,36:72,:]
    allWalk = tensor[:,85:169,:]
    fiveRun = tensor[:,36:41,:]
    fiveWalk = tensor[:,85:90,:]
    allWalkbalancing = tensor[:,170:181,:]
    allWalkuneven = tensor[:,182:218,:]
    boxing = tensor[:,0,:].reshape(45,1,128)
    golfing = tensor[:,9,:].reshape(45,1,128)
    idle = tensor[:,15,:].reshape(45,1,128)
    jump = tensor[:,18,:].reshape(45,1,128)
    run = tensor[:,36,:].reshape(45,1,128)
    shoot = tensor[:,72,:].reshape(45,1,128)
    sit = tensor[:,78,:].reshape(45,1,128)
    sweepfloor = tensor[:,79,:].reshape(45,1,128)
    walk = tensor[:,85,:].reshape(45,1,128)
    walkbalancing = tensor[:,170,:].reshape(45,1,128)
    walkuneven = tensor[:,182,:].reshape(45,1,128)
    washwindow = tensor[:,219,:].reshape(45,1,128)
    fiveBalancing = tensor[:,170:175,:]
    fiveUneven = tensor[:,182:187,:]

    sequence = []
    labelsStacked = []
    if name == 'all':
        sequence = tensor
        labelsStacked = labels
    if name == 'oneEach':
        sequence = np.hstack((boxing, golfing, idle, jump, run, shoot, sit, sweepfloor, walk, walkbalancing, walkuneven, washwindow))
        labelsStacked = np.vstack((labels[0], labels[9], labels[15], labels[18], labels[36], labels[72], labels[78], labels[79], labels[85], labels[170], labels[182], labels[219]))
    if name == 'allRunWalk':
        sequence = np.hstack((allRun, allWalk))
        labelsStacked = np.vstack((labels[36:72], labels[85:169]))
    if name == 'fiveRunWalk':
        sequence = np.hstack((fiveRun, fiveWalk))
        labelsStacked = np.vstack((labels[36:41], labels[85:90]))
    if name == 'allRun':
        sequence = allRun
        labelsStacked = labels[36:72]  
    if name == 'allWalk':
        sequence = allWalk
        labelsStacked = labels[85:169]
    if name == 'allMoving':
        sequence = np.hstack((allJump, allRun, allWalk, allWalkbalancing, allWalkuneven))
        labelsStacked = np.vstack((labels[18:35], labels[36:72], labels[85:169], labels[170:181], labels[182:218]))  
    if name == 'fiveBalancedUneven':
        sequence = np.hstack((fiveBalancing, fiveUneven))
        labelsStacked = np.vstack((labels[170:175], labels[182:187]))

    return sequence, labelsStacked

def loadData():
    print('Loading data...')
    os.chdir('../body/data_complete')
    x = glob.glob('*/*.mat')
    seqList = []
    labelList = []
    minNrFrames = math.inf

    for filename in x:
        seq = loadmat(filename)
        label = filename[:filename.find('\\')]
        labelList.append(label)
        W = np.array(seq.get('W'))
        if W.shape[0] == 1:
            W = W[0][0]
        seqList.append(W)
        if (W.shape[0]/3 < minNrFrames):
            minNrFrames = int(W.shape[0]/3)

    seqList = np.array(seqList)
    return seqList, labelList, minNrFrames

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