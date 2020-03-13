import numpy as np
import numpy.matlib
import sys
from scipy.io.matlab import loadmat
from sktensor import dtensor
import matplotlib.pyplot as plt

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
    action_steps = []
    Lengths = []
    nrPerAction = []
    if alignment_classifier or angle_classifier:
        temp_classification_List = []
        classification_list = []
        classification_labels = []
        for x in np.where(labelsStacked[:,0]==5)[0].tolist():
            temp_classification_List.append((5,subSeqList[x]))

        for x in np.where(labelsStacked[:,0]==9)[0].tolist():
            temp_classification_List.append((9,subSeqList[x]))
        # Top line allows multiple entries of same number (i.e. [3,3,5,...]) bottom does not
        indexes = np.random.randint(len(temp_classification_List),size=int(len(temp_classification_List)))
        indexes = np.random.choice(len(temp_classification_List),len(temp_classification_List),replace=False)
        run_classify = []
        walk_classify = []
        for x in indexes:
            classification_list.append(temp_classification_List[x][1])
            if temp_classification_List[x][0] == 5:
                run_classify.append(temp_classification_List[x][1])
            else:
                walk_classify.append(temp_classification_List[x][1])
            classification_labels.append(temp_classification_List[x][0])
        if alignment_classifier:
            Classifiers.alignment_Classification(run_classify, walk_classify)
        if angle_classifier:
            Classifiers.angle_Classification(classification_list,classification_labels)

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
    fig = plt.figure()
    ax = plt.axes()
    # Plot the movement of the z coordinate of the right foot
    for i in range(0, int(len(action_steps))):
        for x in range(0,int(len(action_steps[i]))):
            ax.plot(action_steps[i][x][11][:])
    tensorList = []
    # Get each sequence and reshape them into 45,1,median length in order to horisontally stack them into a tensor of size 45, number of steps, median length
    for i in range(0,len(action_steps)):
        for x in range(0,len(action_steps[i])):
            temp = np.array(action_steps[i][x]).reshape([45,1,int(medianLength)])
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
    Plotting.plotU1(U1)
    Plotting.plotU2(U2, labelsStacked, action_names)
    Plotting.plotU3(U3)
    plt.show()

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
    plt.show()
if SVM_classifier:
    Classifiers.SVM_Classification(U2[:,0:3], labelsStacked)