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
import glob, os, math
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../Testing')

import Spatial

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


#Align all to T-pose
seq0 = np.array([seqList[46][0,:],seqList[46][1,:],seqList[46][2,:]])

for i, seq in enumerate(seqList):
    frameToTake = int(int(seq.shape[0]/3)/minNrFrames)
    newFrames = np.zeros([minNrFrames*3,15])
    for j in range(0, minNrFrames*3, 3):
        newFrames[j] = seq[j*frameToTake]
        newFrames[j+1] = seq[j*frameToTake+1]
        newFrames[j+2] = seq[j*frameToTake+2]
    seqList[i] = newFrames
    index_inner = [0,9,12]
    for frame in range(0,newFrames.shape[0], 3):
        frameShape = np.zeros([3,15])
        frameShape[0,:] = newFrames[frame,:]
        frameShape[1,:] = newFrames[frame+1,:]
        frameShape[2,:] = newFrames[frame+2,:]
        if (i != 46 and frame != 0):
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

for i, seq in enumerate(seqList):
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

print(seqList[0].shape)

print(dtensor(tensor).shape)

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

    plt.show()

animate(tensor[:,0,:])

# Perform HOSVD and return core tensor and U matrices
def svd(tensor):
    # Perform HOSVD with given dimensions
    U, core_S = hosvd(tensor, [tensor.shape[0], tensor.shape[1], tensor.shape[2]])

    # Core model
    core_S = np.array(core_S)

    # U vectors
    U1 = U[0]
    U2 = U[1]
    U3 = U[2]

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

# Plot the U1 matrix
def plotU1(U1):
    # Create colormap from jet preset
    colormap = plt.get_cmap("jet")

    #Figure 1
    fig1 = plt.figure()
    fig1.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1, wspace=None, hspace=None)
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(U1[:,0], U1[:,1], U1[:,2], s=15, c=np.arange(U1.shape[0]), cmap=colormap)
    ax1.set_xlim(1,-1)
    ax1.set_ylim(-1,1)
    ax1.set_zlim(1,-1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    fig1.suptitle("U1 Matrix")
    plt.show()

# Plot the U2 matrix
def plotU2(U2, labelsStacked, action_names):
    #Figure 2
    fig2 = plt.figure(figsize=(10,10))
    fig2.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1, wspace=None, hspace=None)
    ax2 = fig2.add_subplot(111, projection='3d')
    jet = cm.get_cmap('jet',action_names.shape[0]) # Create colormap with action_names size
    numberPlotted = 0
    for i, action in enumerate(action_names): # Iterate over all actions
        # Get all U2 values that have the given action as the label 
        if (U2[labelsStacked[:,0]==i+1,0].shape[0] != 0):
            if  numberPlotted < len(set(labelsStacked[:,0])) / 2:
                markertype = 'o'
            else:  
                markertype = 'd'
            ax2.scatter(U2[labelsStacked[:,0]==i+1,0], U2[labelsStacked[:,0]==i+1,1], U2[labelsStacked[:,0]==i+1,2], marker=markertype, s=10, color=jet(i), label=action[0][0])   
            numberPlotted += 1
    ax2.view_init(90, -90)
    ax2.set_xlim(0.3,-0.25)
    ax2.set_ylim(0.5,-0.2)
    ax2.set_zlim(1,-1)
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    fig2.suptitle("U2 Matrix")
    plt.show()

# Plot the U3 matrix
def plotU3(U3):
    # Create colormap from jet preset
    colormap = plt.get_cmap("jet")

    # Figure 3
    fig3 = plt.figure(figsize=(10,10))
    fig3.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1, wspace=None, hspace=None)
    ax3 = fig3.add_subplot(111, projection='3d')
    # Scatter all points
    ax3.scatter(U3[:,0], U3[:,1], U3[:,2], s=15, c=np.arange(U3.shape[0]), cmap=colormap)    
    # Draw black line between all points
    ax3.plot(U3[:,0], U3[:,1], U3[:,2], c="black", markersize=1)
    # Label with incrementing counter
    for i in range(U3.shape[0]):
        ax3.text(U3[i,0], U3[i,1], U3[i,2], i, size=6)
    ax3.set_xlim(0.12,0.02)
    ax3.set_ylim(-0.15,0.2)
    ax3.set_zlim(0.2,-0.15)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    fig3.suptitle("U3 Matrix")
    
    plt.show()

# Load data from .mat file
mat = loadmat('../data_tensor.mat')

# Extract previously saved data
labels = np.array(mat.get('labels')) # 225 entries
action_names = np.array(mat.get('action_names')) # 12 entries

# Load full tensor from matlab file
tensor0 = tensor

# Select a given subset of sequences
if len(sys.argv) > 1:
    sequence, labelsStacked = getSequence(tensor0, labels, sys.argv[1])
else:
    sequence, labelsStacked = getSequence(tensor0, labels)
print(sequence.shape)

if len(sequence) == 0:
    print('Error in sequence result, please choose a valid sequence. [All, oneEach, allRunWalk, fiveRunWalk, allRun, allWalk, allMoving, fiveBalancedUneven]')
    exit()

# Compute mean body shape from given sequences
mean_tensor = createMean(sequence)

# Compute new tensor from sequences and mean shape
Tdiff = dtensor(sequence - mean_tensor)

# Perform HOSVD on tensor to get subspaces
U1,U2,U3,core_S = svd(Tdiff)

# Plot the new U matrices
plotU1(U1)
plotU2(U2, labelsStacked, action_names)
plotU3(U3)