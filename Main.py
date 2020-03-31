import numpy as np
import numpy.matlib
import sys
from scipy.io.matlab import loadmat
from sktensor import dtensor
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import TensorHelpers
import DTWHelpers
import AlignData
import Plotting
import Classifiers

import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
import matplotlib

import scipy.optimize as opti

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
# print("Scipy optimization test")
# outer_opt = lambda x : inner_opt(x) - 3
# inner_opt = lambda x : opti.rosen(x)
# x0 = 0.1 * np.arange(10)
# print(x0)
# res = opti.minimize(outer_opt, x0, tol=1e-6)
# resu = opti.rosen([0.99999995, 0.99999992, 0.99999986, 0.99999976, 0.99999955, 0.99999914, 0.99999831, 0.99999665, 0.99999331, 0.99998662])
# print(resu)
# print(resu-3)

# print(res.x)

# raise ValueError()

seqList, labelList, minNrFrames, medianNrFrames = TensorHelpers.loadData()


if True:
    # seqList = AlignData.spatial(seqList)

    # Load data from .mat file
    mat = loadmat('body/data_tensor.mat')

    # Extract previously saved data
    labels = np.array(mat.get('labels')) # 225 entries
    # print("Imported list: ",labels)
    # print("Our List: ",labelList)
    action_names = np.array(mat.get('action_names')) # 12 entries

    # Load full tensor from matlab file
    tensor0 = []
    alignment_classifier = False
    angle_classifier = False
    SVM_classifier = False
    # Select which classifier to run
    if len(sys.argv) > 1:
        if "Alignment" in sys.argv[1]:
            print("Alignement classification requested")
            alignment_classifier = True
        if "Angle" in sys.argv[1]:
            print("Angle classification requested")
            angle_classifier = True
        if "SVM" in sys.argv[1]:
            print("SVM classification requested")
            SVM_classifier = True

    # Select a given subset of sequences
    # if len(sys.argv) > 1:
    #     subSeqList, labelsStacked = TensorHelpers.getSequence(seqList, labels, sys.argv[1])
    # else:
    subSeqList, labelsStacked = TensorHelpers.getSequence(seqList, labels)
    print("Result of get Sequence shape: ", len(subSeqList), len(subSeqList[1]))

    if len(subSeqList) == 0:
        print('Error in sequence result, please choose a valid sequence. [All, oneEach, allRunWalk, fiveRunWalk, allRun, allWalk, allMoving, fiveBalancedUneven]')
        exit()
    # Align the given subset spatially
    subSeqList = AlignData.spatial(subSeqList)
    # Reshape the data to 45, number of frames
    subSeqList = DTWHelpers.reshapeTo45(subSeqList)

    # fig = plt.figure()
    # plt.plot(DTWHelpers.getSyntheticGraph(5), c='b')
    # plt.plot(DTWHelpers.getSyntheticGraph(9), c='g')
    # plt.plot(DTWHelpers.getSyntheticGraph(0), c='r')
    # plt.show()
    currentlyTestingId = 50
    newSeq = subSeqList[currentlyTestingId]
    newSeqLabel = labelsStacked[currentlyTestingId]
    # labelsStacked = np.delete(labelsStacked,0).reshape([119,1])

    seqsTrain, seqsTest, labelsTrain, labelsTest = train_test_split(subSeqList, labelsStacked, test_size = 0.20)

    seqFolds, labelFolds = Classifiers.create_Folds(seqsTrain,labelsTrain,5)

    action_steps = []
    Lengths = []
    nrPerAction = []
    if alignment_classifier:
        print()
        print("Starting Alignment Classifier")
        accuracies = []
        runtimes = []
        for i,x in enumerate(seqFolds):
            seq_list, label_list = Classifiers.getFoldSubList(seqFolds, labelFolds, i)
            accuracy, runtime = Classifiers.alignment_Classification(seq_list, x, label_list, labelFolds[i])
            accuracies.append(accuracy)
            runtimes.append(runtime)
        accuracy , runtime =Classifiers.alignment_Classification(seq_list, seqsTest, label_list, labelsTest)
        accuracies.append(accuracy)
        runtimes.append(runtime)
        meanAccuracy = np.mean(np.array(accuracies))
        meanRunTime = np.mean(np.array(runtimes))
        print("Mean accuracy of alignment classifier: ", meanAccuracy)
        print("Mean runtime of alignment classifier: ", meanRunTime)

    if angle_classifier:
        print()
        print("Starting Angle Classifier")
        angles = []
        accuracies = []
        runtimes = []
        for i, x in enumerate(seqFolds):
            seq_list, label_list = Classifiers.getFoldSubList(seqFolds, labelFolds, i)
            accuracy , splitAngle, runtime = Classifiers.angle_Classification(seq_list, x, label_list, labelFolds[i])
            angles.append(splitAngle)
            runtimes.append(runtime)
            accuracies.append(accuracy)
        meanAngle = np.mean(np.array(angles))
        meanAccuracy = np.mean(np.array(accuracies))
        meanRunTime = np.mean(np.array(runtimes))
        print("Mean Angle of angle classifier",meanAngle)
        print("Mean Runtime of angle classifier",meanRunTime)
        print("Mean Accuracy of angle classifier",meanAccuracy)
        accuracy, angle, runtime = Classifiers.angle_Classification([],seqsTest, [], labelsTest, meanAngle)
        print("Accuracy of final test run using mean angle: ", accuracy)
        print()
        
    # Need to re write so findSteps is called seperately from multiDTW so we can get medianlength and use this length to create our refseq
    for i, action in enumerate(action_names):
        # Select all sequences belonging to the current action.
        if (len(np.where(labelsStacked[:,0]==i+1)[0]) > 0):
            print('Now aligning: ',action_names[i][0][0])
            steps = []
            # Get the synthetic graph for the current action (Only run and walk)
            refSeq = DTWHelpers.getSyntheticGraph(i+1)
            # Create a new list of each sequence used to find the steps and align them
            for x in np.where(labelsStacked[:,0]==i+1)[0].tolist():
                steps.append(subSeqList[x])
            # Find the steps in each sequence and align them to the reference graph
            steps, lengthList = DTWHelpers.multiDTW(steps,11, refSeq)
            # Put the lengths of all steps into a list
            Lengths.extend(lengthList)
            # Create a list containing the id of each action and how many steps belong to that action ([5,41],[9,156] with running and walking for example)
            nrPerAction.append([i+1,len(lengthList)])
            # Put the found steps for the current action into a combined list
            action_steps.append(steps)
    Lengths = np.array(Lengths)
    # Find the median length of the steps
    medianLength = np.median(Lengths)-1
    # fig = plt.figure()
    # ax = plt.axes()

    # # Plot the movement of the z coordinate of the right foot
    # for i in range(0, int(len(action_steps))):
    #     for x in range(0,int(len(action_steps[i]))):
    #         ax.plot(action_steps[i][x][11][:])
    tensorList = []
    # Get each sequence and reshape them into 45,1,median length in order to horisontally stack them into a tensor of size 45, number of steps, median length
    for i in range(0,len(action_steps)):
        for x in range(0,len(action_steps[i])):
            temp = np.array(action_steps[i][x]).reshape([45,1,int(94)])
            #temp = np.array(action_steps[i][x]).reshape([45,1,int(medianLength)])
            if len(tensorList) == 0:
                tensorList = temp
            else:
                tensorList = np.hstack((tensorList,temp))
    tensor = dtensor(tensorList)

    # Compute mean body shape from given sequences
    mean_tensor = TensorHelpers.createMean(tensor)

    # Compute new tensor from sequences and mean shape
    Tdiff = dtensor(tensor - mean_tensor)

    # Perform HOSVD on tensor to get subspaces
    U1,U2,U3,core_S = TensorHelpers.svd(Tdiff)

    # Create a new labellist that looks like the old one from the matlab file, but for the steps in stead of for the full sequences
    labelsStacked = []
    for i in range(0,len(nrPerAction)):
        for x in range(0,nrPerAction[i][1]):
            labelsStacked.append([nrPerAction[i][0]])
    labelsStacked = np.array(labelsStacked)
    
    # Plot the new U matrices
    # Plotting.plotU1(U1)
    # Plotting.plotU2(U2, labelsStacked, action_names)
    # Plotting.plotU3(U3)
    # plt.show()

else:

    seqList = AlignData.temporalLazy(seqList, medianNrFrames)
    seqList = AlignData.spatial(seqList)

    tensor = AlignData.createTensor(seqList, 45, medianNrFrames)

    animation = Plotting.animate(tensor[:,120,:])
    plt.show()

    # Load data from .mat file
    mat = loadmat('body/data_tensor.mat')

    # Extract previously saved data
    labels = np.array(mat.get('labels')) # 225 entries
    action_names = np.array(mat.get('action_names')) # 12 entries

    # Load full tensor from matlab file
    tensor0 = tensor

    # Select a given subset of sequences
    if len(sys.argv) > 1:
        tensor0, labelsStacked = TensorHelpers.getSequence(tensor0, labels, sys.argv[1])
    else:
        tensor0, labelsStacked = TensorHelpers.getSequence(tensor0, labels)
    print(tensor0.shape)

    if len(tensor0) == 0:
        print('Error in sequence result, please choose a valid sequence. [All, oneEach, allRunWalk, fiveRunWalk, allRun, allWalk, allMoving, fiveBalancedUneven]')
        exit()

    #Align Actions in tensor - maybe make loop parallel????
    for i, action in enumerate(action_names):
        if (tensor0[:,labelsStacked[:,0]==i+1,:].shape[1] > 0):
            print('Now aligning: ',action_names[i][0][0])
            tensor0[:,labelsStacked[:,0]==i+1,:] = DTWHelpers.multiDTW(tensor0[:,labelsStacked[:,0]==i+1,:],8)

    ani2 = Plotting.animate(tensor[:,4,:])
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,5))
    for i in range(0, tensor0.shape[1]):
        plot1 = np.zeros([tensor0[:,i,id].shape[0],2])
        plot2 = np.zeros([tensor0[:,i,id].shape[0],2])
        for j in range(0,plot1.shape[0]):
            plot1[j][0] = j
            plot1[j][1] = tensor0[:,i, id, j]
            # plot1[j][1] = angle(AlignedSeqs[i, 33:36, j],AlignedSeqs[i, 30:33, j], AlignedSeqs[i, 27:30, j]) #Arm
            # plot1[j][1] = angle(AlignedSeqs[i, 9:12, j],AlignedSeqs[i, 6:9, j], AlignedSeqs[i, 3:6, j]) #Leg
            plot2[j][0] = j
            plot2[j][1] = tensor0[:,i, j]
        
        ax1.plot(plot1[:,0], plot1[:,1], label=i)    
        ax2.plot(plot2[:,0], plot2[:,1], label=i)

    plt.legend()
    plt.show()


    # Compute mean body shape from given sequences
    mean_tensor = TensorHelpers.createMean(tensor0)

    # Compute new tensor from sequences and mean shape
    Tdiff = dtensor(tensor0 - mean_tensor)

    # Perform HOSVD on tensor to get subspaces
    U1,U2,U3,core_S = TensorHelpers.svd(Tdiff)

    # Plot the new U matrices
    Plotting.plotU1(U1)
    Plotting.plotU2(U2, labelsStacked, action_names)
    Plotting.plotU3(U3)
    # plt.show()
if SVM_classifier:
    # stepSeqs = []
    # # Find all steps in each sequence and put them in a new list
    # finalFirst, finalLast = DTWHelpers.findSteps(np.array(newSeq[:][11][:]))
    # print(finalFirst, finalLast)
    # for j, x in enumerate(finalFirst):
    #     temp = np.array(newSeq)
    #     stepSeqs.append([newSeqLabel, temp[:,x:finalLast[j]]])
    stepSeqs, _ = DTWHelpers.multiDTW([newSeq], 11, DTWHelpers.getSyntheticGraph(0)) 
    from scipy.optimize import rosen_der
    mean_new_shape = np.mean(tensor, axis=(2,1))
    mean_new_shape.reshape(45,1)
    mean_body = np.zeros((45,tensor.shape[2]))
    from datetime import datetime
    for i, x in enumerate(mean_new_shape):
        mean_body[i] = np.resize(x, tensor.shape[2])

    print("U2 mean shape: ",np.mean(U2, axis=0).shape)
    rand_U2 = np.random.rand(U2.shape[0])
    rand_U3 = np.random.rand(U3.shape[0],U3.shape[1])
    import concurrent.futures

    f_hatU2 = lambda u2,U3 : np.add(np.tensordot(np.tensordot(np.tensordot(core_S, U1, (0,1)), u2, (0,0)), U3, (0,1)), mean_body)
    f_hatU3 = lambda u2,u3 : np.add(np.tensordot(np.tensordot(np.tensordot(core_S, U1, (0,1)), u3, (1,0)), u2, (0,0)), mean_new_shape)
    
    # EstU3 = U3
    # M3 = np.transpose(np.tensordot(np.tensordot(core_S, U1, (0,1)), U2[currentlyTestingId],(0,0)))
    # M3pinv = np.linalg.pinv(M3)
    # for i, x in enumerate(EstU3):
    #     EstU3[i] = np.matmul(M3pinv,np.subtract(f_hatU3(U2[currentlyTestingId],x), mean_new_shape))
            
    # M2List = []
    # for i, x in enumerate(U3):
    #     M2List.append(np.transpose(np.tensordot(np.tensordot(core_S, U1, (0,1)),x,(1,0))))
    # M2 = None
    # for i, x in enumerate(M2List):
    #     if M2 is None:
    #         M2 = x
    #     else:
    #         M2 = np.vstack((M2, x))
    # f_hatList = []
    # for i, x in enumerate(U3):
    #     f_hatList.append((np.subtract(f_hatU3(U2[currentlyTestingId],x), mean_new_shape)).reshape(45,1))
    # f_hat_matrix = None
    # for i, x in enumerate(f_hatList):
    #     if f_hat_matrix is None:
    #         f_hat_matrix = x
    #     else:
    #         f_hat_matrix = np.vstack((f_hat_matrix, x))
    # print(M2.shape)
    # print(f_hat_matrix.shape)
    # print(np.tensordot(core_S, U1, (0,1)).shape)
    # EstU2 = np.matmul(np.linalg.pinv(M2),f_hat_matrix).reshape(U2.shape[0])

    # print("Difference in U3", np.linalg.norm(U3 - EstU3))
    # print("Difference in U2", np.linalg.norm(U2[currentlyTestingId] - EstU2))
    print(np.min(U2))
    print(np.max(U2))
    for g, seq in enumerate(stepSeqs):
        init_U2 = np.mean(U2, axis=0)
        u2_hat = np.random.uniform(np.min(U2), high=np.max(U2), size=U2.shape[0])#rand_U2
        init_U3 = np.mean(U3, axis=0)
        U3_hat = np.random.uniform(np.min(U3), high=np.max(U3), size=(U3.shape[0],U3.shape[1]))#rand_U3
        U2_list = []
        errors = []
        # def minimize_U3(id):
        #     opt_funU3 = lambda u3: 0.5 * np.abs(np.linalg.norm(f_hatU3(u2_hat,u3) - seq[:,id]))**2
        #     return (id,opti.minimize(opt_funU3, U3_hat[id,:], method='SLSQP', jac=rosen_der, options={'maxiter':30}).x)#, constraints = ({'type': 'eq', 'fun': lambda x: x.sum() - 1.0, 'jac': lambda x: np.ones_like(x)})).x)
        for j in range (0,20):
            start = datetime.now()
            # if j == 0:
            #     with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            #         fibSubmit = {executor.submit(minimize_U3, n): n for n in range(0, seq.shape[1])}

            #         for future in concurrent.futures.as_completed(fibSubmit):
            #             try:
            #                 n, f = future.result()
            #             except Exception as exc:
            #                 print("Error! {0}".format(exc))
            #             else:
            #                 U3_hat[n] = f
                # with concurrent.futures.ThreadPoolExecutor() as executor:
                #     for i, out1 in enumerate(executor.map(minimize_U3, range(0, x.shape[1]))):
                #         U3_hat[i] = out1
                # for i in range (0,x.shape[1]):
                #     opt_funU3 = lambda u3: 0.5 * np.abs(np.linalg.norm(f_hatU3(u2_hat,u3) - x[:,i]))**2
                #     # print(opt_fun(U3[i,:]))
                #     U3_hat[i] = opti.minimize(opt_funU3, U3_hat[i,:], method='Nelder-Mead', options={'maxiter':15}).x


            
            M3 = np.transpose(np.tensordot(np.tensordot(core_S, U1, (0,1)), u2_hat,(0,0)))
            M3pinv = np.linalg.pinv(M3)
            for i, x in enumerate(U3_hat):
                U3_hat[i] = np.matmul(M3pinv,np.subtract(f_hatU3(u2_hat,x), mean_new_shape))
                

            M2List = []
            for i, x in enumerate(U3_hat):
                M2List.append(np.transpose(np.tensordot(np.tensordot(core_S, U1, (0,1)),x,(1,0))))
            M2 = None
            for i, x in enumerate(M2List):
                if M2 is None:
                    M2 = x
                else:
                    M2 = np.vstack((M2, x))
            f_hatList = []
            for i, x in enumerate(U3_hat):
                f_hatList.append((np.subtract(f_hatU3(u2_hat,x), mean_new_shape)).reshape(45,1))
            f_hat_matrix = None
            for i, x in enumerate(f_hatList):
                if f_hat_matrix is None:
                    f_hat_matrix = x
                else:
                    f_hat_matrix = np.vstack((f_hat_matrix, x))
            u2_hat = np.matmul(np.linalg.pinv(M2),f_hat_matrix).reshape(U2.shape[0])
            #u2_hat = np.mean(u2_hat, axis=0)
            end = datetime.now()
            errors.append(0.5 * np.abs(np.linalg.norm(f_hatU2(u2_hat,U3_hat) - seq))**2)
            U2_list.append(u2_hat)

            print("M2 condition", np.linalg.cond(M2))
            print("M3 condition", np.linalg.cond(M3))
            print("U3_hat condition", np.linalg.cond(U3_hat))
            print("U2 condition", np.linalg.cond(U2))
            print("U3 condition", np.linalg.cond(U3))
            print("Test condition", np.linalg.cond(np.add(np.tensordot(np.tensordot(np.tensordot(core_S, U1, (0,1)), U2[currentlyTestingId], (0,0)), U3, (0,1)), mean_body)))
            print("Iteration:", j, " Time taken to estimate: ", end-start)

        newMatrix1 = np.tensordot(core_S, U1, (0,1))
        newMatrix2 = np.tensordot(newMatrix1,u2_hat, (0,0))
        newMatrix = np.tensordot(newMatrix2,U3_hat,(0,1))
        FirstFrameModel = np.add(newMatrix, mean_body)

        xs = []
        ys = []
        zs = []

        # Split the data into x,y,z coordinates for each frame
        for frm in range(0, FirstFrameModel.shape[1]):
            for j in range(0, 45, 3):
                xs.append(FirstFrameModel[j][frm])
                ys.append(FirstFrameModel[j+1][frm])
                zs.append(FirstFrameModel[j+2][frm])

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
        ani = animation.FuncAnimation(fig, update, FirstFrameModel.shape[1], fargs=(xs,ys,zs), interval=200)
        plt.show()

    # opt_funU2 = lambda u2: 0.5 * np.abs(np.linalg.norm(f_hatU2(u2,U3_hat) - x))**2
    # u2_hat = opti.minimize(opt_funU2, u2_hat, method='Newton-CG', jac=rosen_der, options={'maxiter':10}).x
    # U2_list[g,:] = u2_hat
    
        #print(np.array(u2_hat).shape)
        print("Coordinates of U2: ", u2_hat.reshape(1,U2.shape[0])[0,:3])
        labelsStacked[currentlyTestingId] = [14]
        newU2 = U2
        newLabels = labelsStacked
        for x in U2_list:
            newU2 = np.vstack((newU2, x.reshape(1,U2.shape[0])))
            newLabels = np.vstack((newLabels, [13]))
        Plotting.plotU2(newU2, newLabels, np.append(np.append(action_names, 'Test'), 'Original'))
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(errors)

        plt.show()
    # opt_fun = lambda u2,u3: f_hat(u2,u3) - 
    sys.exit()


    print()
    print("Starting SVM Classifier")
    seqsTrain, seqsTest, labelsTrain, labelsTest = train_test_split(U2[:,0:3], labelsStacked, test_size = 0.20)
    
    seqFolds, labelFolds = Classifiers.create_Folds(seqsTrain,labelsTrain,5)
    accuracies = []
    runtimes = []
    for i, x in enumerate(seqFolds):
            seq_list, label_list = Classifiers.getFoldSubList(seqFolds, labelFolds, i)
            accuracy, runtime = Classifiers.SVM_Classification(np.array(seq_list), np.array(x), np.array(label_list), np.array(labelFolds[i]))
            accuracies.append(accuracy)
            runtimes.append(runtime)
    accuracy, runtime = Classifiers.SVM_Classification(seqsTrain, seqsTest, labelsTrain, labelsTest)
    accuracies.append(accuracy)
    runtimes.append(runtime)
    print("Mean SVM accuracy: ", np.mean(np.array(accuracies)))
    print("Mean SVM runtime: ", np.mean(np.array(runtimes)))