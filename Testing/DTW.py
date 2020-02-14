from dtw import *
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

import mpl_toolkits.mplot3d as plt3d
import matplotlib.animation as animation

from fastdtw import fastdtw
from dtaidistance import dtw as dtw2
from dtaidistance import dtw_visualisation as dtwvis


from scipy.io.matlab import loadmat

import Spatial

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

seq1 = loadmat('../body/data_complete/run/09_07.mat')
seq2 = loadmat('../body/data_complete/run/16_36.mat')
seq3 = loadmat('../body/data_complete/walk/08_02.mat')
W1 = np.array(seq1.get('W'))[0][0]
W2 = np.array(seq2.get('W'))[0][0]
W3 = np.array(seq3.get('W'))[0][0]
print(W1.shape)
print(W2.shape)
print(W3.shape)

seq0 = None
def NewLength(seq, NmFrames, seq0):
    frameToTake = int(int(seq.shape[0]/3)/NmFrames)
    idx = np.round(np.linspace(0,int(seq.shape[0]/3)-1,NmFrames)).astype(int)
    newFrames = np.zeros([NmFrames*3,15])
    for j, i in enumerate(idx):
        newFrames[j*3] = seq[i*3]
        newFrames[j*3+1] = seq[i*3+1]
        newFrames[j*3+2] = seq[i*3+2]
    index_inner = [0,9,12]
    for frame in range(0,newFrames.shape[0], 3):
        frameShape = np.zeros([3,15])
        frameShape[0,:] = newFrames[frame,:]
        frameShape[1,:] = newFrames[frame+1,:]
        frameShape[2,:] = newFrames[frame+2,:]
        if (seq0 is None):
            seq0 = frameShape
        else:
            _, _, transform = Spatial.procrustes(np.transpose(seq0[:,[0,7,9,12]]), np.transpose(frameShape[:,[0,7,9,12]]), False, True)
            Z = np.matmul(transform['scale']*np.transpose(frameShape),transform['rotation'])
            frameShape = np.transpose(Z)
            triangle_static = seq0[:,index_inner]
            triangle_deform = frameShape[:,index_inner]
            _,_, transform2 = Spatial.procrustes(np.transpose(triangle_static), np.transpose(triangle_deform), False, True)
            frameShape_transformed = np.matmul(transform2['scale']*np.transpose(frameShape),transform2['rotation'])
            frameShape = np.transpose(frameShape_transformed)
            newFrames[frame,:] = frameShape[0,:]
            newFrames[frame+1,:] = frameShape[1,:]
            newFrames[frame+2,:] = frameShape[2,:]
    return newFrames, seq0

W1, seqSp = NewLength(W1, 128, seq0)
seq0 = seqSp
W2, seqSp = NewLength(W2, 128, seq0)
W3, seqSp = NewLength(W3, 128, seq0)

W1copy = np.zeros([45, int(W1.shape[0]/3)])
for j in range(0,int(W1.shape[0]/3)):
    k = 3*j
    shape = W1[k:k+3,:]
    points = np.zeros([45])
    for l in range(0,15):
        points[l*3] = shape[0][l]
        points[(l*3)+2] = shape[1][l]
        points[(l*3)+1] = shape[2][l]
    W1copy[:,j] = points


W2copy = np.zeros([45, int(W2.shape[0]/3)])
for j in range(0,int(W2.shape[0]/3)):
    k = 3*j
    shape = W2[k:k+3,:]
    points = np.zeros([45])
    for l in range(0,15):
        points[l*3] = shape[0][l]
        points[(l*3)+2] = shape[1][l]
        points[(l*3)+1] = shape[2][l]
    W2copy[:,j] = points


W3copy = np.zeros([45, int(W3.shape[0]/3)])
for j in range(0,int(W3.shape[0]/3)):
    k = 3*j
    shape = W3[k:k+3,:]
    points = np.zeros([45])
    for l in range(0,15):
        points[l*3] = shape[0][l]
        points[(l*3)+2] = shape[1][l]
        points[(l*3)+1] = shape[2][l]
    W3copy[:,j] = points

print(W1copy.shape)
print(W2copy.shape)
print(W3copy.shape)


W1Foot = W1copy[11,:]
W2Foot = W2copy[11,:]
W3Foot = W3copy[11,:]

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

fig = plt.figure()
ax = plt.axes()
ax.plot(W1plot[:,0], W1plot[:,1],c="b")
ax.plot(W2plot[:,0],W2plot[:,1],c="g")
ax.plot(W3plot[:,0],W3plot[:,1],c="r")

plt.show()


distance, path = fastdtw(W1plot, W2plot, dist=euclidean, radius=15)
NpPath = np.array(path)

WSmasked = [W1Foot,W2Foot]

distance, paths = dtw2.warping_paths(W3Foot, W1Foot, window=13)
print(distance)
best = dtw2.best_path(paths)
dtwvis.plot_warpingpaths(W3Foot, W1Foot, paths,best)
# print(best)
NpPath = np.array(best)

distance, paths = dtw2.warping_paths(W2Foot, W1Foot, window=13)
print(distance)
best = dtw2.best_path(paths)
dtwvis.plot_warpingpaths(W2Foot, W1Foot, paths,best)
# print(best)
#NpPath = np.array(best)

plt.show()
# distance = dtw2.distance_matrix_fast(WSmasked)
# print(distance)
# #print(NpPath)

# #print(NpPath)

W1Aligned = np.zeros([45,W2plot.shape[0]])
for i in range(0,W2plot.shape[0]):
    W1Aligned[:,i] = W1copy[:,int(NpPath[i][1])]
# print("distance: ",distance)
# print(path)

W3Aligned = np.zeros([45,W2plot.shape[0]])
for i in range(0,W2plot.shape[0]):
    W3Aligned[:,i] = W3copy[:,int(NpPath[i][0])]


# alignment = dtw(W1Foot, W2Foot, keep_internals=True)

# ## Display the warping curve, i.e. the alignment curve
# alignment.plot(type="threeway")

# ## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
# dtw(W1Foot, W2Foot, keep_internals=True, 
#     step_pattern=rabinerJuangStepPattern(6, "c"))\
#     .plot(type="twoway",offset=-2)

def animate(array):
    xs = []
    ys = []
    zs = []

    print(array.shape)
    nfr = int(array.shape[1])

    # Split the data into x,y,z coordinates for each frame
    for j in range(0, array.shape[1]):
        for i in range(0, array.shape[0], 3):
            xs.append(array[i][j])
            ys.append(array[i+1][j])
            zs.append(array[i+2][j])
            
    # Create plot and empty points to update
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sct, = ax.plot([], [], [], "o", markersize=2)

    # Update function to be called each frame
    def update(ifrm, xa, ya, za):
        # Clear all lines (except points)
        ax.lines = [ax.lines[0]]

        sct.set_data(xa[ifrm*15:ifrm*15+14], ya[ifrm*15:ifrm*15+14])
        sct.set_3d_properties(za[ifrm*15:ifrm*15+14])
        # For drawing the lines between points
        # Right leg
        ax.plot3D([xa[ifrm*15+0], xa[ifrm*15+1]], [ya[ifrm*15+0], ya[ifrm*15+1]], [za[ifrm*15+0], za[ifrm*15+1]], 'steelblue', markersize=2)
        ax.plot3D([xa[ifrm*15+1], xa[ifrm*15+2]], [ya[ifrm*15+1], ya[ifrm*15+2]], [za[ifrm*15+1], za[ifrm*15+2]], 'steelblue', markersize=2)
        ax.plot3D([xa[ifrm*15+2], xa[ifrm*15+3]], [ya[ifrm*15+2], ya[ifrm*15+3]], [za[ifrm*15+2], za[ifrm*15+3]], 'steelblue', markersize=2)
        # Left leg
        ax.plot3D([xa[ifrm*15+0], xa[ifrm*15+4]], [ya[ifrm*15+0], ya[ifrm*15+4]], [za[ifrm*15+0], za[ifrm*15+4]], 'steelblue', markersize=2)
        ax.plot3D([xa[ifrm*15+4], xa[ifrm*15+5]], [ya[ifrm*15+4], ya[ifrm*15+5]], [za[ifrm*15+4], za[ifrm*15+5]], 'steelblue', markersize=2)
        ax.plot3D([xa[ifrm*15+5], xa[ifrm*15+6]], [ya[ifrm*15+5], ya[ifrm*15+6]], [za[ifrm*15+5], za[ifrm*15+6]], 'steelblue', markersize=2)
        # Spine
        ax.plot3D([xa[ifrm*15+0], xa[ifrm*15+7]], [ya[ifrm*15+0], ya[ifrm*15+7]], [za[ifrm*15+0], za[ifrm*15+7]], 'steelblue', markersize=2)
        ax.plot3D([xa[ifrm*15+7], xa[ifrm*15+8]], [ya[ifrm*15+7], ya[ifrm*15+8]], [za[ifrm*15+7], za[ifrm*15+8]], 'steelblue', markersize=2)
        # Right arm
        ax.plot3D([xa[ifrm*15+7], xa[ifrm*15+9]], [ya[ifrm*15+7], ya[ifrm*15+9]], [za[ifrm*15+7], za[ifrm*15+9]], 'steelblue', markersize=2)
        ax.plot3D([xa[ifrm*15+9], xa[ifrm*15+10]], [ya[ifrm*15+9], ya[ifrm*15+10]], [za[ifrm*15+9], za[ifrm*15+10]], 'steelblue', markersize=2)
        ax.plot3D([xa[ifrm*15+10], xa[ifrm*15+11]], [ya[ifrm*15+10], ya[ifrm*15+11]], [za[ifrm*15+10], za[ifrm*15+11]], 'steelblue', markersize=2)
        # Left arm
        ax.plot3D([xa[ifrm*15+7], xa[ifrm*15+12]], [ya[ifrm*15+7], ya[ifrm*15+12]], [za[ifrm*15+7], za[ifrm*15+12]], 'steelblue', markersize=2)
        ax.plot3D([xa[ifrm*15+12], xa[ifrm*15+13]], [ya[ifrm*15+12], ya[ifrm*15+13]], [za[ifrm*15+12], za[ifrm*15+13]], 'steelblue', markersize=2)
        ax.plot3D([xa[ifrm*15+13], xa[ifrm*15+14]], [ya[ifrm*15+13], ya[ifrm*15+14]], [za[ifrm*15+13], za[ifrm*15+14]], 'steelblue', markersize=2)


    # Limit coordinates for all axes
    ax.set_xlim(30,-30)
    ax.set_ylim(30,-30)
    ax.set_zlim(-30,30)

    # Set labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Run animation with the update function and point lists
    ani = animation.FuncAnimation(fig, update, nfr, fargs=(xs,ys,zs), interval=1)
    #ani.save('../../animation.gif', writer='imagemagick', fps=30)

    return ani

ani1 = animate(W1Aligned)

ani2 = animate(W3Aligned)

ani1._start
ani2._start
plt.show()