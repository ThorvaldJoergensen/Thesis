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
# #DtaiDistance testing:
# s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
# s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0, 0])
# path = dtw2.warping_path(s1, s2)
# dtwvis.plot_warping(s1, s2, path, filename="warp.png")

# #Fastdtw testing:
# x = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]])
# y = np.array([[2,2], [3,3], [4,4]])
# distance, path = fastdtw(x, y, dist=euclidean)
# print("distance: ",distance)
# print(path)

# fig = plt.figure()
# ax = plt.axes()
# ax.scatter(x[:,0], x[:,1],s=15,c="b")
# ax.scatter(y[:,0],y[:,1],s=15,c="g")
# plt.show()

# #dtw-python Testing:
# idx = np.linspace(0,6.28,num=100)
# query = np.sin(idx) + np.random.uniform(size=100)/10.0

# ## A cosine is for template; sin and cos are offset by 25 samples
# template = np.cos(idx)

# ## Find the best match with the canonical recursion formula
# alignment = dtw(query, template, keep_internals=True)

# ## Display the warping curve, i.e. the alignment curve
# alignment.plot(type="threeway")

# ## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
# dtw(query, template, keep_internals=True, 
#     step_pattern=rabinerJuangStepPattern(6, "c"))\
#     .plot(type="twoway",offset=-2)

# ## See the recursion relation, as formula and diagram
# print(rabinerJuangStepPattern(6,"c"))
# rabinerJuangStepPattern(6,"c").plot()

# seq1 = np.zeros(10)
# seq2 = np.zeros(10)

# for i, x in enumerate(seq1):
#     seq1[i] = i*3

# for i, x in enumerate(seq2):
#     if i+1 < seq1.shape[0]:
#         seq2[i] = seq1[i+1]
#     else:
#         seq2[i] = (i+1)*3

# alignment2 = dtw(seq2, seq1, keep_internals=True)

# alignment2.plot(type="threeway")

# dtw(seq2, seq1, keep_internals=True, 
#     step_pattern=rabinerJuangStepPattern(6, "c"))\
#     .plot(type="twoway",offset=-2)
id = 8

seqList, labels, minNrFrames, medianNrFrames = Helpers.loadData()
seqList = AlignData.temporalLazy(seqList, medianNrFrames)
print(seqList.shape)
runSeqs = seqList[36:72]
walkSeqs = seqList[85:169]
maxRunFrames = 0
for i in runSeqs:
    if int(i.shape[0]/3) > maxRunFrames:
        maxRunFrames = int(i.shape[0]/3)
maxWalkFrames = 0
for i in walkSeqs:
    if int(i.shape[0]/3) > maxWalkFrames:
        maxWalkFrames = int(i.shape[0]/3)
print(maxWalkFrames)
print(maxRunFrames)
framesToAlignTo = min(maxWalkFrames, maxRunFrames)
Aligned = np.array([runSeqs[3],runSeqs[24],walkSeqs[2],walkSeqs[42]])
Aligned = runSeqs
# Aligned = walkSeqs
# Aligned = np.concatenate((walkSeqs, runSeqs))

Aligned = AlignData.spatial(Aligned)
AlignedRightForm = []
RightFormFull = []
for i, x in enumerate(Aligned):
    W1copy = np.zeros([45, int(x.shape[0]/3)])
    for j in range(0,int(x.shape[0]/3)):
        k = 3*j
        shape = x[k:k+3,:]
        points = np.zeros([45])
        for l in range(0,15):
            points[l*3] = shape[0][l]
            points[(l*3)+2] = shape[1][l]
            points[(l*3)+1] = shape[2][l]
        W1copy[:,j] = points
    AlignedRightForm.append(W1copy[id,:])
    RightFormFull.append(W1copy)
AlignedRightForm = np.array(AlignedRightForm)
RightFormFull = np.array(RightFormFull)
def scaleSeqs (W3Foot, W4Foot):
    maxIValue = np.max(W3Foot)
    maxJValue = np.max(W4Foot)
    newISeqs = np.zeros(W3Foot.shape)
    newJSeqs = np.zeros(W4Foot.shape)

    if (abs(maxIValue) > abs(maxJValue*2)):
            print("Doing it 1 ", maxIValue, maxJValue)
            dividerValue = maxIValue / maxJValue
            dividerValue = dividerValue * (3/4)
            newISeqs = W3Foot
            newJSeqs = W4Foot + (maxIValue / 2)
            print("old 2, ", np.max(W4Foot))
            print("new 2, ", np.max(newJSeqs))
            print("Divider ", dividerValue)
    elif (abs(maxJValue) > abs(maxIValue*2)): 
            print("doing it 2 ", maxIValue, maxJValue)
            dividerValue = maxJValue / maxIValue
            dividerValue = dividerValue * (3/4)
            newISeqs = W3Foot + (maxJValue / 2)
            newJSeqs = W4Foot
            print("Old 1, " ,np.max(W3Foot))
            print("New 1, ", np.max(newISeqs))
            print("divider ", dividerValue)
    else:
            print("not doing it ", maxIValue, maxJValue)
            newISeqs = W3Foot
            newJSeqs = W4Foot
    return newISeqs,newJSeqs

def multiDTW(seqs, id):
    aligned = []
    for i in seqs:
        aligned.append(i[id,:])
    aligned = np.array(aligned)
    #print(dtw2.distance_matrix_fast(aligned, parallel=True))
    sim_matrix = np.zeros([seqs.shape[0],seqs.shape[0]])
    iPos = 0
    JPos = 0
    firstI = 0
    firstJ = 0
    minDist = math.inf
    firstPath = []
    for i, x in enumerate(aligned):
        for j, y in enumerate(aligned):
            if i == j:
                sim_matrix[i][j] = math.inf
            else:
                distance = dtwalign(aligned[j],x, window_type="sakoechiba", window_size=int(min(aligned[JPos].shape[0], x.shape[0])/10), dist_only=True).distance
                sim_matrix[i][j] = distance
                if distance < minDist:
                    minDist = distance
                    firstI = i
                    firstJ = j
                    iPos = i
                    JPos = j

    print("Id's: ", iPos, JPos)
    print(sim_matrix)
    AlignedSeqs = np.zeros([aligned.shape[0],45,aligned[iPos].shape[0]])
    AlignedSeqs[iPos,:,:] = seqs[iPos]
    AlignedIds = [iPos]
    aligned = np.array(aligned)
    for g in range(0,aligned.shape[0]-1):
        # newISeqs, newJSeqs = scaleSeqs(aligned[JPos], AlignedSeqs[iPos, id, :])
        # _,paths = dtw2.warping_paths(aligned[JPos], AlignedSeqs[iPos, id, :], window=int(min(aligned[JPos].shape[0], AlignedSeqs[iPos,id,:].shape[0])/10))
        # res = dtwalign(newJSeqs, newISeqs, dist='matching')
        res = dtwalign(aligned[JPos], AlignedSeqs[iPos, id, :], step_pattern="typeIds", window_size=int(min(aligned[JPos].shape[0], x.shape[0])/10))
        path = res.get_warping_path(target="query")
        # path = np.array(dtw2.best_path(paths))
        JAligned = np.zeros([45,AlignedSeqs[iPos].shape[1]])
        JAligned = seqs[JPos][:, path]
        # for j in range(0,AlignedSeqs[iPos].shape[1]):
        #     JAligned[:,j] = seqs[JPos][:,int(path[j][0])]
        AlignedSeqs[JPos,:,:] = JAligned
        AlignedIds.append(JPos)
        if g != aligned.shape[0]-2:
            newSim = np.zeros([len(AlignedIds),aligned.shape[0]])
            for f,x in enumerate(AlignedIds):
                newSim[f] = sim_matrix[x]
            minDist = math.inf
            for i,x in enumerate(newSim):
                for j,d in enumerate(newSim[i]):
                    if j in AlignedIds or i == j:
                        continue
                    else:
                        if newSim[i][j] < minDist:
                            minDist = newSim[i][j]
                            iPos = AlignedIds[i]
                            JPos = j
    return AlignedSeqs
    
aligned = multiDTW(np.array(RightFormFull), id)


AlignedSeqs = aligned#np.concatenate((runAligned, walkAligned))

print(AlignedSeqs.shape)

animation = Plotting.animate(AlignedSeqs[0])

animation2 = Plotting.animate(AlignedSeqs[1])
plt.show()

print("Second part")
print(runSeqs.shape)
print(walkSeqs.shape)
# seq1 = loadmat('run/09_07.mat')
# seq2 = loadmat('run/16_36.mat')
# seq3 = loadmat('walk/08_02.mat')
# W1 = np.array(seq1.get('W'))[0][0]
# W2 = np.array(seq2.get('W'))[0][0]
# W3 = np.array(seq3.get('W'))[0][0]
# print(W1.shape)
# print(W2.shape)
# print(W3.shape)

# seq0 = None
# def NewLength(seq, NmFrames, seq0):
#     frameToTake = int(int(seq.shape[0]/3)/NmFrames)
#     idx = np.round(np.linspace(0,int(seq.shape[0]/3)-1,NmFrames)).astype(int)
#     newFrames = np.zeros([NmFrames*3,15])
#     for j, i in enumerate(idx):
#         newFrames[j*3] = seq[i*3]
#         newFrames[j*3+1] = seq[i*3+1]
#         newFrames[j*3+2] = seq[i*3+2]
#     index_inner = [0,9,12]
#     for frame in range(0,newFrames.shape[0], 3):
#         frameShape = np.zeros([3,15])
#         frameShape[0,:] = newFrames[frame,:]
#         frameShape[1,:] = newFrames[frame+1,:]
#         frameShape[2,:] = newFrames[frame+2,:]
#         if (seq0 is None):
#             seq0 = frameShape
#         else:
#             _, _, transform = Helpers.procrustes(np.transpose(seq0[:,[0,7,9,12]]), np.transpose(frameShape[:,[0,7,9,12]]), False, True)
#             Z = np.matmul(transform['scale']*np.transpose(frameShape),transform['rotation'])
#             frameShape = np.transpose(Z)
#             triangle_static = seq0[:,index_inner]
#             triangle_deform = frameShape[:,index_inner]
#             _,_, transform2 = Helpers.procrustes(np.transpose(triangle_static), np.transpose(triangle_deform), False, True)
#             frameShape_transformed = np.matmul(transform2['scale']*np.transpose(frameShape),transform2['rotation'])
#             frameShape = np.transpose(frameShape_transformed)
#             newFrames[frame,:] = frameShape[0,:]
#             newFrames[frame+1,:] = frameShape[1,:]
#             newFrames[frame+2,:] = frameShape[2,:]
#     return newFrames, seq0

# W1, seqSp = NewLength(W1, 128, seq0)
# seq0 = seqSp
# W2, seqSp = NewLength(W2, 128, seq0)
# W3, seqSp = NewLength(W3, 128, seq0)

# W1copy = np.zeros([45, int(W1.shape[0]/3)])
# for j in range(0,int(W1.shape[0]/3)):
#     k = 3*j
#     shape = W1[k:k+3,:]
#     points = np.zeros([45])
#     for l in range(0,15):
#         points[l*3] = shape[0][l]
#         points[(l*3)+2] = shape[1][l]
#         points[(l*3)+1] = shape[2][l]
#     W1copy[:,j] = points


# W2copy = np.zeros([45, int(W2.shape[0]/3)])
# for j in range(0,int(W2.shape[0]/3)):
#     k = 3*j
#     shape = W2[k:k+3,:]
#     points = np.zeros([45])
#     for l in range(0,15):
#         points[l*3] = shape[0][l]
#         points[(l*3)+2] = shape[1][l]
#         points[(l*3)+1] = shape[2][l]
#     W2copy[:,j] = points


# W3copy = np.zeros([45, int(W3.shape[0]/3)])
# for j in range(0,int(W3.shape[0]/3)):
#     k = 3*j
#     shape = W3[k:k+3,:]
#     points = np.zeros([45])
#     for l in range(0,15):
#         points[l*3] = shape[0][l]
#         points[(l*3)+2] = shape[1][l]
#         points[(l*3)+1] = shape[2][l]
#     W3copy[:,j] = points

# print(W1copy.shape)
# print(W2copy.shape)
# print(W3copy.shape)


W1Foot = AlignedSeqs[0, id, :]
W2Foot = AlignedSeqs[1, id, :]
W3Foot = AlignedSeqs[2, id, :]
W4Foot = AlignedSeqs[3, id, :]

def angle(a,b,c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


W1plot = np.zeros([W1Foot.shape[0],2])
for i in range(0,W1plot.shape[0]):
    W1plot[i][0] = i
    W1plot[i][1] = W1Foot[i]
W2plot = np.zeros([W2Foot.shape[0],2])
for i in range(0,W2plot.shape[0]):
    W2plot[i][0] = i
    W2plot[i][1] = W2Foot[i]
W3plot = np.zeros([W3Foot.shape[0],2])
for i in range(0,W3plot.shape[0]):
    W3plot[i][0] = i
    W3plot[i][1] = W3Foot[i]
W4plot = np.zeros([W4Foot.shape[0],2])
for i in range(0,W4plot.shape[0]):
    W4plot[i][0] = i
    W4plot[i][1] = W4Foot[i]

testArray = []

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,5))
for i in range(0, AlignedSeqs.shape[0]):
    plot1 = np.zeros([AlignedSeqs[i,id].shape[0],2])
    plot2 = np.zeros([AlignedSeqs[i,id].shape[0],2])
    for j in range(0,plot1.shape[0]):
        plot1[j][0] = j
        plot1[j][1] = AlignedSeqs[i, id, j]
        # plot1[j][1] = angle(AlignedSeqs[i, 33:36, j],AlignedSeqs[i, 30:33, j], AlignedSeqs[i, 27:30, j]) #Arm
        # plot1[j][1] = angle(AlignedSeqs[i, 9:12, j],AlignedSeqs[i, 6:9, j], AlignedSeqs[i, 3:6, j]) #Leg
        plot2[j][0] = j
        plot2[j][1] = AlignedRightForm[i, j]
    
    ax1.plot(plot1[:,0], plot1[:,1], label=i)    
    ax2.plot(plot2[:,0], plot2[:,1], label=i)

    testArray.append(np.nanmin(plot1[:,1]))

plt.legend()
plt.show()

print(np.max(np.array(testArray)))
# for i, x in enumerate(plot1[:,1]):

    

# W1plot = np.zeros([W1Foot.shape[0], 2])
# print("tasdlka", AlignedSeqs[0, 33:35, :].shape)
# for i in range(0, W1plot.shape[0]):
#     W1plot[i][0] = i
#     W1plot[i][1] = angle(AlignedSeqs[0, 33:36, i],AlignedSeqs[0, 30:33, i], AlignedSeqs[0, 27:30, i])

# W2plot = np.zeros([W2Foot.shape[0],2])
# for i in range(0,W2plot.shape[0]):
#     W2plot[i][0] = i
#     W2plot[i][1] = angle(AlignedSeqs[1, 33:36, i],AlignedSeqs[1, 30:33, i], AlignedSeqs[1, 27:30, i])

# W3plot = np.zeros([W3Foot.shape[0],2])
# for i in range(0,W3plot.shape[0]):
#     W3plot[i][0] = i
#     W3plot[i][1] = angle(AlignedSeqs[2, 33:36, i],AlignedSeqs[2, 30:33, i], AlignedSeqs[2, 27:30, i])

# W4plot = np.zeros([W4Foot.shape[0],2])
# for i in range(0,W4plot.shape[0]):
#     W4plot[i][0] = i
#     W4plot[i][1] = angle(AlignedSeqs[3, 33:36, i],AlignedSeqs[3, 30:33, i], AlignedSeqs[3, 27:30, i])

# fig = plt.figure()
# ax = plt.axes()
# ax.plot(W1plot[:,0], W1plot[:,1],c="b", label="0")
# ax.plot(W2plot[:,0],W2plot[:,1],c="g", label="1")
# ax.plot(W3plot[:,0],W3plot[:,1],c="r", label="2")
# ax.plot(W4plot[:,0],W4plot[:,1],c="purple", label="3")
# plt.legend()

# W1Foot = AlignedRightForm[0, :]
# W2Foot = AlignedRightForm[1, :]
# W3Foot = AlignedRightForm[2, :]
# W4Foot = AlignedRightForm[3, :]

# W1plot = np.zeros([W1Foot.shape[0],2])
# for i in range(0,W1plot.shape[0]):
#     W1plot[i][0] = i
#     W1plot[i][1] = W1Foot[i]
# W2plot = np.zeros([W2Foot.shape[0],2])
# for i in range(0,W2plot.shape[0]):
#     W2plot[i][0] = i
#     W2plot[i][1] = W2Foot[i]
# W3plot = np.zeros([W3Foot.shape[0],2])
# for i in range(0,W3plot.shape[0]):
#     W3plot[i][0] = i
#     W3plot[i][1] = W3Foot[i]
# W4plot = np.zeros([W4Foot.shape[0],2])
# for i in range(0,W4plot.shape[0]):
#     W4plot[i][0] = i
#     W4plot[i][1] = W4Foot[i]

# # W1plot = np.zeros([W1Foot.shape[0],2])
# # for i in range(0,W1plot.shape[0]):
# #     W1plot[i][0] = i
# #     W1plot[i][1] = angle(RightFormFull[0, 9:12, i],RightFormFull[0, 6:9, i], RightFormFull[0, 3:6, i])

# # W2plot = np.zeros([W2Foot.shape[0],2])
# # for i in range(0,W2plot.shape[0]):
# #     W2plot[i][0] = i
# #     W2plot[i][1] = angle(RightFormFull[1, 9:12, i],RightFormFull[1, 6:9, i], RightFormFull[1, 3:6, i])

# # W3plot = np.zeros([W3Foot.shape[0],2])
# # for i in range(0,W3plot.shape[0]):
# #     W3plot[i][0] = i
# #     W3plot[i][1] = angle(RightFormFull[2, 9:12, i],RightFormFull[2, 6:9, i], RightFormFull[2, 3:6, i])

# # W4plot = np.zeros([W4Foot.shape[0],2])
# # for i in range(0,W4plot.shape[0]):
# #     W4plot[i][0] = i
# #     W4plot[i][1] = angle(RightFormFull[3, 9:12, i],RightFormFull[3, 6:9, i], RightFormFull[3, 3:6, i])

# fig = plt.figure()
# ax = plt.axes()
# ax.plot(W1plot[:,0], W1plot[:,1],c="b", label="0")
# ax.plot(W2plot[:,0],W2plot[:,1],c="g", label="1")
# ax.plot(W3plot[:,0],W3plot[:,1],c="r", label="2")
# ax.plot(W4plot[:,0],W4plot[:,1],c="purple", label="3")
# plt.legend()
# plt.show()


res = dtwalign(W3plot, W2plot, window_type="sakoechiba",step_pattern="typeIds")
res.plot_path()

res = dtwalign(W2Foot, W1Foot, window_type="sakoechiba",step_pattern="typeIds")
res.plot_path()

res = dtwalign(W3Foot, W4Foot, window_type="sakoechiba",step_pattern="typeIds")
res.plot_path()
# NpPath = np.array(path)

# WSmasked = [W1Foot,W2Foot]
# distance, paths = dtw2.warping_paths(W3Foot, W2Foot)
# print(distance)
# best = dtw2.best_path(paths)
# dtwvis.plot_warpingpaths(W3Foot, W2Foot, paths,best)
# # print(best)
# NpPath = np.array(best)
# distance, paths = dtw2.warping_paths(W2Foot, W1Foot)
# print(distance)
# best = dtw2.best_path(paths)
# dtwvis.plot_warpingpaths(W2Foot, W1Foot, paths,best)

# distance, paths = dtw2.warping_paths(W3Foot, W4Foot)
# print(distance)
# best = dtw2.best_path(paths)
# dtwvis.plot_warpingpaths(W3Foot, W4Foot, paths,best)
# print(best)
#NpPath = np.array(best)

plt.show()

# for i, seq in enumerate(RightFormFull[:61]):
#     print(i)

    
#     animation1 = Plotting.animate(AlignedSeqs[i])
#     animation1._start

#     animation2 = Plotting.animate(RightFormFull[i])
#     animation2._start

#     plt.show()