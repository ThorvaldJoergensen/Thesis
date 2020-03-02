import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import sys
import mpl_toolkits.mplot3d as plt3d
import matplotlib.animation as animation
import math

from fastdtw import fastdtw
from dtaidistance import dtw as dtw2
from dtaidistance import dtw_visualisation as dtwvis
from dtwalign import dtw as dtwalign


from scipy.io.matlab import loadmat

import Helpers
sys.path.insert(1, '../')
import Plotting
sys.path.insert(1,'../Tensor')
import AlignData

id = 11

seqList, labels, minNrFrames, medianNrFrames = Helpers.loadData()
# seqList = AlignData.temporalLazy(seqList, medianNrFrames)
print(seqList.shape)
runSeqs = seqList[36:72]
walkSeqs = seqList[85:169]

ShortestRun = runSeqs[3]
print(len(ShortestRun))
print(len(ShortestRun[1]))
ShortestWalk = walkSeqs[22]
print(ShortestRun[0:3][:][0][1])
ShortestRunNP = np.zeros([45,int(len(ShortestRun)/3)])
for j in range(0,int(len(ShortestRun)/3)):
    k = 3*j
    shape = ShortestRun[k:k+3][:]
    points = np.zeros([45])
    for l in range(0,15):
        points[l*3] = shape[0][l]
        points[(l*3)+2] = shape[1][l]
        points[(l*3)+1] = shape[2][l]
    ShortestRunNP[:,j] = points

ShortestWalkNP = np.zeros([45,int(len(ShortestWalk)/3)])
for j in range(0,int(len(ShortestWalk)/3)):
    k = 3*j
    shape = ShortestWalk[k:k+3][:]
    points = np.zeros([45])
    for l in range(0,15):
        points[l*3] = shape[0][l]
        points[(l*3)+2] = shape[1][l]
        points[(l*3)+1] = shape[2][l]
    ShortestWalkNP[:,j] = points

ShortestRunOneId = ShortestRunNP[id,:]
ShortestWalkOneId = ShortestWalkNP[id,:]

print(ShortestRunOneId.shape)
print(ShortestWalkOneId.shape)

RunPlot = np.zeros([ShortestRunOneId.shape[0],2])
WalkPlot = np.zeros([ShortestWalkOneId.shape[0],2])
# not necessary
for i in range(0,RunPlot.shape[0]):
    RunPlot[i][0] = i
    RunPlot[i][1] = ShortestRunOneId[i]
for j in range(0,WalkPlot.shape[0]):
    WalkPlot[j][0] = j
    WalkPlot[j][1] = ShortestWalkOneId[j]

fig = plt.figure()
ax = plt.axes()
ax.plot(RunPlot[:,0], RunPlot[:,1],c="b", label="Run")
ax.plot(WalkPlot[:,0],WalkPlot[:,1],c="g", label="Walk")
plt.show()