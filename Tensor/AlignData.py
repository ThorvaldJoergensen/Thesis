import numpy as np
import sys
from sktensor import dtensor
sys.path.insert(1, '../Testing')
import Helpers


def spatial(data):
    print('Starting spacial alignment')
    seq0 = None
    for seq in data:
        index_inner = [0,9,12]
        for frame in range(0,seq.shape[0], 3):
            frameShape = np.zeros([3,15])
            frameShape[0,:] = seq[frame,:]
            frameShape[1,:] = seq[frame+1,:]
            frameShape[2,:] = seq[frame+2,:]
            if (seq0 is None):
                seq0 = frameShape
            else:
                _, _, transform = Helpers.procrustes(np.transpose(seq0[:,[0,7,9,12]]), np.transpose(frameShape[:,[0,7,9,12]]), False, True)
                Z = np.matmul(transform['scale']*np.transpose(frameShape),transform['rotation'])
                frameShape = np.transpose(Z)
                triangle_static = seq0[:,index_inner]
                triangle_deform = frameShape[:,index_inner]
                _,_, transform2 = Helpers.procrustes(np.transpose(triangle_static), np.transpose(triangle_deform), False, True)
                frameShape_transformed = np.matmul(transform2['scale']*np.transpose(frameShape),transform2['rotation'])
                frameShape = np.transpose(frameShape_transformed)
                seq[frame,:] = frameShape[0,:]
                seq[frame+1,:] = frameShape[1,:]
                seq[frame+2,:] = frameShape[2,:]
    print('Finished spacial alignment')
    return data

def temporalLazy(data, minNrFrames):
    print('Starting lazy temporal alignment')
    for i, seq in enumerate(data):
        idx = np.round(np.linspace(0,int(seq.shape[0]/3)-1,minNrFrames)).astype(int)
        newFrames = np.zeros([minNrFrames*3,15])
        for j, k in enumerate(idx):
            newFrames[j*3] = seq[k*3]
            newFrames[j*3+1] = seq[k*3+1]
            newFrames[j*3+2] = seq[k*3+2]
        data[i] = newFrames
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