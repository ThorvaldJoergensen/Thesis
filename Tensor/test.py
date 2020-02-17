import numpy as np
import sys
from scipy.io.matlab import loadmat
from sktensor import dtensor, cp_als
from sktensor.tucker import hosvd
import mpl_toolkits.mplot3d as plt3d
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm
import glob, os, math

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

print(minNrFrames)
print(np.array(seqList)[0].shape)
print(np.array(labelList).shape)

tensor = np.zeros([45, 225, minNrFrames])

testList = []
for i, seq in enumerate(seqList):
    frameToTake = int(int(seq.shape[0]/3)/minNrFrames)
    newFrames = np.zeros([minNrFrames*3,15])
    for j in range(0, minNrFrames*3, 3):
        newFrames[j] = seq[j*frameToTake]
        newFrames[j+1] = seq[j*frameToTake+1]
        newFrames[j+2] = seq[j*frameToTake+2]
    testList.append(newFrames)


for i, seq in enumerate(testList):
    W3 = np.zeros([45, minNrFrames])
    for j in range(0, minNrFrames):
        k = 3*j
        shape = seq[k:k+3,:]
        points = np.zeros([45])
        for l in range(0, 15):
            points[l*3] = shape[0][l]
            points[(l*3)+2] = shape[1][l]
            points[(l*3)+1] = shape[2][l]
        W3[:, j] = points
    tensor[:, i,:] = W3


tensor = dtensor(tensor)

# U, S = hosvd(tensor, [45, 225, 128])

# core_S = np.array(S)
# U1 = U[0]
# U2 = U[1]
# U3 = U[2]

# rowU2 = U2[0]
# newMatrix = np.tensordot(core_S, U1, (0,1))
# newMatrix = np.tensordot(newMatrix, rowU2, (0,0))
# newMatrix = np.tensordot(newMatrix, U3, (0,1))

# tensor = dtensor(newMatrix)

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
    fig.suptitle("Aligned")

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

    return ani

def animate2(array):
    xs = []
    ys = []
    zs = []
    
    nfr = int(array.shape[0]/3)

    # Split the data into x,y,z coordinates for each frame
    for i in range(0, array.shape[0], 3):
        for j in range(0, 15):
            xs.append(array[i][j])
            ys.append(array[i+2][j])
            zs.append(array[i+1][j])
            
    # Create plot and empty points to update
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sct, = ax.plot([], [], [], "o", markersize=2)
    fig.suptitle("Original")

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
    ani = animation.FuncAnimation(fig, update, nfr, fargs=(xs,ys,zs), interval=0.01)

    return ani

def animate3(array1, array2):
    xs1 = []
    ys1 = []
    zs1 = []
    xs2 = []
    ys2 = []
    zs2 = []

    nfr = int(array1.shape[1])

    # Split the data into x,y,z coordinates for each frame
    for j in range(0, array1.shape[1]):
        for i in range(0, array1.shape[0], 3):
            xs1.append(array1[i][j])
            ys1.append(array1[i+1][j])
            zs1.append(array1[i+2][j])
            
    # Create plot and empty points to update
    fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')
    ax[0] = fig.add_subplot(111, projection='3d')
    ax[1] = fig.add_subplot(111, projection='3d')
    sct, = ax[0].plot([], [], [], "o", markersize=2)
    sct, = ax[1].plot([], [], [], "o", markersize=2)
    

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

    plt.show()

for i, seq in enumerate(seqList[113:]):
    print(i+113)

    
    animation1 = animate(tensor[:,i+113,:])
    animation1._start

    animation2 = animate2(seq)
    animation2._start

    plt.show()