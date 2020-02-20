import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import matplotlib.animation as animation
import numpy as np
from matplotlib import cm

def animate(array, filepath=''):
    xs = []
    ys = []
    zs = []

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
    if filepath != '':
        ani.save(filepath, writer='imagemagick', fps=30)
    ani._start

    return ani

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
    # for i in range(U3.shape[0]):
    #     ax3.text(U3[i,0], U3[i,1], U3[i,2], i, size=6)
    ax3.set_xlim(0.12,0.02)
    ax3.set_ylim(-0.15,0.2)
    ax3.set_zlim(0.2,-0.15)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    fig3.suptitle("U3 Matrix")
    