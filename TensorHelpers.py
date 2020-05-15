import numpy as np
from sktensor import dtensor
from sktensor.tucker import hosvd
import glob, os, math
from scipy.io.matlab import loadmat

# Compute mean shape tensor from given tensor subselection
def createMean(tensor):
    mean_new_shape = np.mean(tensor, axis=(2,1))
    mean_new_shape.reshape(45,1)
    mean_tensor = np.zeros((45,tensor.shape[1],tensor.shape[2]))
    for i, x in enumerate(mean_new_shape):
        mean_tensor[i] = np.resize(x, (tensor.shape[1],tensor.shape[2]))
    return mean_tensor

# Selects a given predefined subset of sequences
def getSequence(tensor, labels, name = 'allRunWalk'):
    allJump = tensor[18:35]
    allRun = tensor[36:72]
    allWalk = tensor[85:170]
    fiveRun = tensor[36:41]
    fiveWalk = tensor[85:90]
    allWalkbalancing = tensor[170:181]
    allWalkuneven = tensor[182:218]
    boxing = tensor[0]
    golfing = tensor[9]
    idle = tensor[15]
    jump = tensor[18]
    run = tensor[36]
    shoot = tensor[72]
    sit = tensor[78]
    sweepfloor = tensor[79]
    walk = tensor[85]
    walkbalancing = tensor[170]
    walkuneven = tensor[182]
    washwindow = tensor[219]
    fiveBalancing = tensor[170:175]
    fiveUneven = tensor[182:187]

    sequence = []
    labelsStacked = []
    # Stack selected action into a uniform shape again
    if name == 'all':
        sequence = tensor
        labelsStacked = labels
    if name == 'oneEach':
        sequence = [boxing, golfing, idle, jump, run, shoot, sit, sweepfloor, walk, walkbalancing, walkuneven, washwindow]
        labelsStacked = np.vstack((labels[0], labels[9], labels[15], labels[18], labels[36], labels[72], labels[78], labels[79], labels[85], labels[170], labels[182], labels[219]))
    if name == 'allRunWalk':
        sequence = np.concatenate((allRun,allWalk))
        labelsStacked = np.vstack((labels[36:72], labels[85:170]))
    if name == 'fiveRunWalk':
        sequence = np.concatenate((fiveRun, fiveWalk))
        labelsStacked = np.vstack((labels[36:41], labels[85:90]))
    if name == 'allRun':
        sequence = allRun
        labelsStacked = labels[36:72]  
    if name == 'allWalk':
        sequence = allWalk
        labelsStacked = labels[85:169]
    if name == 'allMoving':
        sequence = np.concatenate((allJump, allRun, allWalk, allWalkbalancing, allWalkuneven))
        labelsStacked = np.vstack((labels[18:35], labels[36:72], labels[85:170], labels[170:181], labels[182:218]))  
    if name == 'fiveBalancedUneven':
        sequence = np.concatenate((fiveBalancing, fiveUneven))
        labelsStacked = np.vstack((labels[170:175], labels[182:187]))

    return sequence, labelsStacked

def loadData():
    print('Loading data...')
    x = glob.glob('body/data_complete/*/*.mat')
    seqList = []
    labelList = []
    minNrFrames = math.inf
    sizes = []
    for filename in x:
        seq = loadmat(filename)
        label = filename[:filename.find('\\')]
        labelList.append(label)
        W = np.array(seq.get('W'))
        if W.shape[0] == 1:
            W = W[0][0]
        seqList.append(W)
        sizes.append(W.shape[0]/3)
        if (W.shape[0]/3 < minNrFrames):
            minNrFrames = int(W.shape[0]/3)

    seqList = np.array(seqList)
    medianNrFrames = np.median(sizes)
    return seqList, labelList, minNrFrames, int(medianNrFrames)

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

    print('Finished HOSVD')
    return U1, U2, U3, core_S