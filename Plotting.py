import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import matplotlib.animation as animation
import numpy as np
from matplotlib import cm

# Create animation plot of a sequence with shape (45, x) where x is the number of frames
def animate(array, filepath=''):
    xs = []
    ys = []
    zs = []

    # Keep count of number of frames in the sequence
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

        # Set x, y & z values 
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
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Create animation with the update function and point lists
    ani = animation.FuncAnimation(fig, update, nfr, fargs=(xs,ys,zs), interval=1)

    # If filepath is given save the animation as a gif (imagemagick installation is required)
    if filepath != '':
        ani.save(filepath, writer='imagemagick', fps=30)

    # Start the animation
    ani._start

    # Return animation object
    return ani

# Plot the U1 matrix
def plotU1(U1):
    # Create colormap from jet preset
    colormap = plt.get_cmap("jet")

    # Create figure
    fig = plt.figure()
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1, wspace=None, hspace=None)
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points 
    ax.scatter(U1[:,0], U1[:,1], U1[:,2], s=15, c=np.arange(U1.shape[0]), cmap=colormap)

    # Limit x, y & z axis 
    ax.set_xlim(1,-1)
    ax.set_ylim(-1,1)
    ax.set_zlim(1,-1)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set title on plot
    fig.suptitle("U1 Matrix")

# Plot the U2 matrix
def plotU2(U2, labelsStacked, action_names):
    # Create figure
    fig = plt.figure(figsize=(10,10))
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1, wspace=None, hspace=None)
    ax = fig.add_subplot(111, projection='3d')

    # Create colormap with the amount of actions from jet preset
    jet = cm.get_cmap('jet',action_names.shape[0]) 

    actionsPlotted = 0

    # Iterate over all actions
    for i, action in enumerate(action_names): 
        # Get all U2 values that have the given action as the label 
        if (U2[labelsStacked[:,0]==i+1,0].shape[0] != 0):
            # Switch markertype when half of the actions have been plotted
            if  actionsPlotted < len(set(labelsStacked[:,0])) / 2:
                markertype = 'o'
            else:  
                markertype = 'd'
            
            # Plot the points with the color and label based on action
            ax.scatter(U2[labelsStacked[:,0]==i+1,0], U2[labelsStacked[:,0]==i+1,1], U2[labelsStacked[:,0]==i+1,2], marker=markertype, s=10, color=jet(i), label=action[0][0])   
            actionsPlotted += 1
    # Set view angle
    ax.view_init(90, -90)
    
    # Limit x, y & z axis 
    ax.set_xlim(0.3,-0.25)
    ax.set_ylim(0.5,-0.2)
    ax.set_zlim(1,-1)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set title on plot
    fig.suptitle("U2 Matrix")
    
    # Show labels
    ax.legend()

# Plot the U3 matrix
def plotU3(U3):
    # Create colormap from jet preset
    colormap = plt.get_cmap("jet")

    # Create figure
    fig = plt.figure(figsize=(10,10))
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1, wspace=None, hspace=None)
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points 
    ax.scatter(U3[:,0], U3[:,1], U3[:,2], s=15, c=np.arange(U3.shape[0]), cmap=colormap)    

    # Draw black line between points
    ax.plot(U3[:,0], U3[:,1], U3[:,2], c="black", markersize=1)

    # Label with incrementing counter
    # for i in range(U3.shape[0]):
    #     ax3.text(U3[i,0], U3[i,1], U3[i,2], i, size=6)
    
    # Limit x, y & z axis 
    ax.set_xlim(0.12,0.02)
    ax.set_ylim(-0.15,0.2)
    ax.set_zlim(0.2,-0.15)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set title on plot
    fig.suptitle("U3 Matrix")
    