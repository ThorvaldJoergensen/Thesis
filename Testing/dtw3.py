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

turnSeq = walkSeqs[35]
# turnSeq = runSeqs[3]
turnSeq2 = walkSeqs[36]

ShortestRun = runSeqs[3]
print(len(ShortestRun)/3)
ShortestWalk = walkSeqs[22]
print(len(ShortestWalk)/3)
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

# fig = plt.figure()
# ax = plt.axes()
# ax.plot(RunPlot[:,0], RunPlot[:,1],c="b", label="Run")
# ax.plot(WalkPlot[:,0],WalkPlot[:,1],c="g", label="Walk")
# plt.show()

turnSeqNP = np.zeros([45,int(len(turnSeq)/3)])
turnSeq2NP = np.zeros([45,int(len(turnSeq2)/3)])
for j in range(0,int(len(turnSeq)/3)):
    k = 3*j
    shape = turnSeq[k:k+3][:]
    points = np.zeros([45])
    for l in range(0,15):
        points[l*3] = shape[0][l]
        points[(l*3)+2] = shape[1][l]
        points[(l*3)+1] = shape[2][l]
    turnSeqNP[:,j] = points
for j in range(0,int(len(turnSeq2)/3)):
    k = 3*j
    shape = turnSeq2[k:k+3][:]
    points = np.zeros([45])
    for l in range(0,15):
        points[l*3] = shape[0][l]
        points[(l*3)+2] = shape[1][l]
        points[(l*3)+1] = shape[2][l]
    turnSeq2NP[:,j] = points
# ani = Plotting.animate(turnSeqNP)
# ani._start
turnSeqOneId = turnSeqNP[id,:]
turnSeq2OneId = turnSeq2NP[id,:]
fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(turnSeqOneId, c="r")
ax2.plot(turnSeq2OneId, c = "y")
plt.show()

y1 = turnSeq2OneId

shift = int(y1.shape[0]/100)
print("Shift: ",shift)
window_size = int(y1.shape[0]/10)
print("Window Size: ", window_size)
threshold = np.var(y1)
chuncklist = []
avg_y1 = np.average(y1)
print(avg_y1)
print("Variance af hele lortet: ",np.var(y1))
while True:
    for i in range(0,y1.shape[0],shift):
        y1_window = y1[i:(i+window_size)]
        variance = np.var(y1_window)
        if variance >= threshold:
            pass_counter = 0
            for x in y1_window:
                if x < avg_y1 and (pass_counter == 0 or pass_counter == 2):
                    pass_counter = pass_counter +1
                elif x > avg_y1 and pass_counter == 1:
                    pass_counter = pass_counter +1
            if pass_counter >= 3:
                chuncklist.append((i,y1_window))
    if len(chuncklist) >= 1:
        print(window_size)
        break
    if window_size >= y1.shape[0]:
        raise Exception("lorrrrrrt")
    window_size = window_size+shift

FirstofChunk = []
LastofChunk = []
for x in chuncklist:
    FirstofChunk.append(x[0])
    LastofChunk.append(x[0]+window_size)
    # print(x)
print(FirstofChunk)
print(LastofChunk)
FinalFirst = []
FinalLast = []

for i, val in enumerate(FirstofChunk):
    if i != 0 and FirstofChunk[i-1] >= val-shift:
        continue
    else:
        FinalFirst.append(val)

for i, val in enumerate(LastofChunk):
    if i < len(LastofChunk)-1 and LastofChunk[i+1] <= val+shift:
        continue
    else:
        FinalLast.append(val)
popIndexes = []
for i, val in enumerate(FinalFirst):
    for j in range(0,i):
        if FinalLast[j] > val and val > FinalFirst[i-1]+((FinalLast[j] - FinalFirst[i-1])/2):
            FinalFirst[i] = FinalLast[j]
        elif val < FinalFirst[i-1]+((FinalLast[j] - FinalFirst[i-1])/2):
            FinalFirst.pop(i)
            FinalLast.pop(i)
print(FinalFirst)
print(FinalLast)
fig = plt.figure()
ax = plt.axes()
ax.plot(y1,c="b", label="1")
ax.scatter(FinalFirst,np.full([len(FinalFirst)],-33), c="g")
ax.scatter(FinalLast,np.full([len(FinalLast)],-33), c="r")
# ax.plot(y22[:,0],y22[:,1],c="g", label="2")
plt.show()


var1 = np.var(turnSeq2OneId[200:500])
var2 = np.var(turnSeq2OneId[500:700])
print(var1)
print(var2)

turnSeqNPLastPart = turnSeqOneId[600:]
turnSeq2NPLastPart = turnSeq2OneId[700:]
res = dtwalign(turnSeqNPLastPart, turnSeq2NPLastPart,step_pattern="asymmetric", open_begin=True)
res.plot_path()
print(res.path)

y22path = res.path[:,0]
y12path = res.path[:,1]
# y12path = res.get_warping_path(target="reference")

plt.plot(turnSeqNPLastPart[y22path], c="g")
plt.plot(turnSeq2NPLastPart[y12path],c="b")
plt.show()

turnseqoneAligned = turnSeqNP[:,599+res.path[:,0]]
turnseq2Aligned = turnSeq2NP[:,699+res.path[:,1]]

ani2 = Plotting.animate(turnseqoneAligned)
ani3 = Plotting.animate(turnseq2Aligned)
ani2._start
ani3._start
plt.show()
