import numpy as np
import numpy.matlib
import sys
from scipy.io.matlab import loadmat
from sktensor import dtensor, cp_als
from sktensor.tucker import hosvd
import mpl_toolkits.mplot3d as plt3d
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm
import glob, os, math, sys
sys.path.insert(1, '../Testing')
import Spatial


os.chdir('../body/data_complete')
x = glob.glob('*/*.mat')

seqList = []
labelList = []
minNrFrames = math.inf
seq_origin = np.array(loadmat("run/09_10.mat").get('W')[0][0])

seq = np.array([seq_origin[3], seq_origin[4], seq_origin[5]])

print(seq.shape)

seq2 = np.array([seq_origin[12]+10, seq_origin[13], seq_origin[14]])

print(seq2.shape)

index_inner = [0,7,9,12]
_, _, transform = Spatial.procrustes(np.transpose(seq[:,[0,7,9,12]]), np.transpose(seq2[:,[0,7,9,12]]), False, True)
Z = np.matmul(np.transpose(seq2), transform['rotation']) + np.matlib.repmat(transform['translation'],15,1)
seq2 = np.transpose(Z)
triangle_static = seq[:,index_inner]
triangle_deform = seq2[:,index_inner]
_,_, transform2 = Spatial.procrustes(np.transpose(triangle_static), np.transpose(triangle_deform), False, True)
seq2_transformed = np.matmul(np.transpose(seq2), transform2['rotation']) + np.matlib.repmat(transform2['translation'],15,1)
seq2 = np.transpose(seq2_transformed)

print(np.linalg.norm(seq-seq2))

xs = []
ys = []
zs = []

# Split the data into x,y,z coordinates for each frame
for i in range(0, seq.shape[0], 3):
    for j in range(0, 15):
        xs.append(seq[i][j])
        ys.append(seq[i+2][j])
        zs.append(seq[i+1][j])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sct, = ax.plot([], [], [], "o", markersize=2)

# sct.set_data(xs[0:ifrm*15+14], ys[0:ifrm*15+14])
# sct.set_3d_properties(zs[0:ifrm*15+14])
# For drawing the lines between points
# Right leg
ax.plot3D([xs[0+0], xs[0+1]], [ys[0+0], ys[0+1]], [zs[0+0], zs[0+1]], 'steelblue', markersize=2)
ax.plot3D([xs[0+1], xs[0+2]], [ys[0+1], ys[0+2]], [zs[0+1], zs[0+2]], 'steelblue', markersize=2)
ax.plot3D([xs[0+2], xs[0+3]], [ys[0+2], ys[0+3]], [zs[0+2], zs[0+3]], 'steelblue', markersize=2)
# Left leg
ax.plot3D([xs[0+0], xs[0+4]], [ys[0+0], ys[0+4]], [zs[0+0], zs[0+4]], 'steelblue', markersize=2)
ax.plot3D([xs[0+4], xs[0+5]], [ys[0+4], ys[0+5]], [zs[0+4], zs[0+5]], 'steelblue', markersize=2)
ax.plot3D([xs[0+5], xs[0+6]], [ys[0+5], ys[0+6]], [zs[0+5], zs[0+6]], 'steelblue', markersize=2)
# Spine
ax.plot3D([xs[0+0], xs[0+7]], [ys[0+0], ys[0+7]], [zs[0+0], zs[0+7]], 'steelblue', markersize=2)
ax.plot3D([xs[0+7], xs[0+8]], [ys[0+7], ys[0+8]], [zs[0+7], zs[0+8]], 'steelblue', markersize=2)
# Right arm
ax.plot3D([xs[0+7], xs[0+9]], [ys[0+7], ys[0+9]], [zs[0+7], zs[0+9]], 'steelblue', markersize=2)
ax.plot3D([xs[0+9], xs[0+10]], [ys[0+9], ys[0+10]], [zs[0+9], zs[0+10]], 'steelblue', markersize=2)
ax.plot3D([xs[0+10], xs[0+11]], [ys[0+10], ys[0+11]], [zs[0+10], zs[0+11]], 'steelblue', markersize=2)
# Left arm
ax.plot3D([xs[0+7], xs[0+12]], [ys[0+7], ys[0+12]], [zs[0+7], zs[0+12]], 'steelblue', markersize=2)
ax.plot3D([xs[0+12], xs[0+13]], [ys[0+12], ys[0+13]], [zs[0+12], zs[0+13]], 'steelblue', markersize=2)
ax.plot3D([xs[0+13], xs[0+14]], [ys[0+13], ys[0+14]], [zs[0+13], zs[0+14]], 'steelblue', markersize=2)


xs = []
ys = []
zs = []

# Split the data into x,y,z coordinates for each frame
for i in range(0, seq2.shape[0], 3):
    for j in range(0, 15):
        xs.append(seq2[i][j])
        ys.append(seq2[i+2][j])
        zs.append(seq2[i+1][j])

# sct.set_data(xs[0:ifrm*15+14], ys[0:ifrm*15+14])
# sct.set_3d_properties(zs[0:ifrm*15+14])
# For drawing the lines between points
# Right leg
ax.plot3D([xs[0+0], xs[0+1]], [ys[0+0], ys[0+1]], [zs[0+0], zs[0+1]], 'green', markersize=2)
ax.plot3D([xs[0+1], xs[0+2]], [ys[0+1], ys[0+2]], [zs[0+1], zs[0+2]], 'green', markersize=2)
ax.plot3D([xs[0+2], xs[0+3]], [ys[0+2], ys[0+3]], [zs[0+2], zs[0+3]], 'green', markersize=2)
# Left leg
ax.plot3D([xs[0+0], xs[0+4]], [ys[0+0], ys[0+4]], [zs[0+0], zs[0+4]], 'green', markersize=2)
ax.plot3D([xs[0+4], xs[0+5]], [ys[0+4], ys[0+5]], [zs[0+4], zs[0+5]], 'green', markersize=2)
ax.plot3D([xs[0+5], xs[0+6]], [ys[0+5], ys[0+6]], [zs[0+5], zs[0+6]], 'green', markersize=2)
# Spine
ax.plot3D([xs[0+0], xs[0+7]], [ys[0+0], ys[0+7]], [zs[0+0], zs[0+7]], 'green', markersize=2)
ax.plot3D([xs[0+7], xs[0+8]], [ys[0+7], ys[0+8]], [zs[0+7], zs[0+8]], 'green', markersize=2)
# Right arm
ax.plot3D([xs[0+7], xs[0+9]], [ys[0+7], ys[0+9]], [zs[0+7], zs[0+9]], 'green', markersize=2)
ax.plot3D([xs[0+9], xs[0+10]], [ys[0+9], ys[0+10]], [zs[0+9], zs[0+10]], 'green', markersize=2)
ax.plot3D([xs[0+10], xs[0+11]], [ys[0+10], ys[0+11]], [zs[0+10], zs[0+11]], 'green', markersize=2)
# Left arm
ax.plot3D([xs[0+7], xs[0+12]], [ys[0+7], ys[0+12]], [zs[0+7], zs[0+12]], 'green', markersize=2)
ax.plot3D([xs[0+12], xs[0+13]], [ys[0+12], ys[0+13]], [zs[0+12], zs[0+13]], 'green', markersize=2)
ax.plot3D([xs[0+13], xs[0+14]], [ys[0+13], ys[0+14]], [zs[0+13], zs[0+14]], 'green', markersize=2)
# Limit coordinates for all axes
ax.set_xlim(30,-30)
ax.set_ylim(30,-30)
ax.set_zlim(-30,30)

# Set labels
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")



plt.show()