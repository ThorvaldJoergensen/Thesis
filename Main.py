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
from datetime import datetime

# Load the data
seqList, labelList, minNrFrames, medianNrFrames = TensorHelpers.loadData()

# Load data from .mat file
mat = loadmat('body/data_tensor.mat')

# Extract previously saved data
labels = np.array(mat.get('labels')) # 225 entries
action_names = np.array(mat.get('action_names')) # 12 entries

# Load full tensor from matlab file
tensor0 = []
# Booleans to switch between classifiers
alignment_classifier = False
angle_classifier = False
SVM_classifier = False
# Select which classifier to run
if len(sys.argv) > 1:
    if "alignment" in sys.argv[1].lower():
        print("Alignement classification requested")
        alignment_classifier = True
    if "angle" in sys.argv[1].lower():
        print("Angle classification requested")
        angle_classifier = True
    if "svm" in sys.argv[1].lower():
        print("SVM classification requested")
        SVM_classifier = True
else: # If none given, run all
    print("Running all classifiers")
    alignment_classifier = True
    angle_classifier = True
    SVM_classifier = True

# Get running and walking sequences
subSeqList, labelsStacked = TensorHelpers.getSequence(seqList, labels)

if len(subSeqList) == 0:
    print('Error in sequence result, please choose a valid sequence. [All, oneEach, allRunWalk, fiveRunWalk, allRun, allWalk, allMoving, fiveBalancedUneven]')
    exit()

# Align the given subset spatially
subSeqList = AlignData.spatial(subSeqList)
# Reshape the data to 45, number of frames
subSeqList = DTWHelpers.reshapeTo45(subSeqList)

# Plot synthetic graphs
# fig = plt.figure()
# plt.plot(DTWHelpers.getSyntheticGraph(5), c='b')
# plt.plot(DTWHelpers.getSyntheticGraph(9), c='g')
# plt.plot(DTWHelpers.getSyntheticGraph(0), c='r')
# plt.show()

# Plot two timeseries of different classes
# fig = plt.figure()
# ax = plt.axes()
# ax.plot(subSeqList[6][11][:],c="b", label="Run")
# ax.plot(subSeqList[63][11][:],c="r", label="Walk")
# plt.legend()
# plt.show()

angle_accuracy = []
alignment_accuracy = []
svm_accuracy = []

for u in range (0,10):
    # Split data into 80% training and 20% testing
    seqsTrain, seqsTest, labelsTrain, labelsTest = train_test_split(subSeqList, labelsStacked, test_size = 0.20)

    # Create 5 folds from the training data
    seqFolds, labelFolds = Classifiers.create_Folds(seqsTrain, labelsTrain, 5)

    # Alignment classifier
    if alignment_classifier:
        print()
        print("Starting Alignment Classifier")
        accuracies = []
        runtimes = []
        # Iterate through the 5 folds
        for i,x in enumerate(seqFolds):
            # Get the sequences and labels from the fold
            seq_list, label_list = Classifiers.getFoldSubList(seqFolds, labelFolds, i)
            # Run the alignment classifier
            accuracy, runtime = Classifiers.alignment_Classification(seq_list, x, label_list, labelFolds[i])
            # Save accuracy and runtime
            accuracies.append(accuracy)
            runtimes.append(runtime)
        # Run the alignment classifier on the test set
        accuracy , runtime = Classifiers.alignment_Classification(seq_list, seqsTest, label_list, labelsTest)
        accuracies.append(accuracy)
        runtimes.append(runtime)
    # Run the alignment classifier on the test set
    accuracy , runtime = Classifiers.alignment_Classification(seq_list, seqsTest, label_list, labelsTest)
    accuracies.append(accuracy)
    runtimes.append(runtime)
    # Compute mean accuracy and runtime
    meanAccuracy = np.mean(np.array(accuracies))
    meanRunTime = np.mean(np.array(runtimes))
    print("Accuracy of test set of alignment classifier: ", accuracy)
    print("Mean accuracy of alignment classifier: ", meanAccuracy)
    print("Mean runtime of alignment classifier: ", meanRunTime)

# Angle classifier
if angle_classifier:
    print()
    print("Starting Angle Classifier")
    angles = []
    accuracies = []
    runtimes = []
    # Iterate through the 5 folds
    for i, x in enumerate(seqFolds):
        # Get the sequences and labels from the fold
        seq_list, label_list = Classifiers.getFoldSubList(seqFolds, labelFolds, i)
        # Run the angle classifier
        accuracy , splitAngle, runtime = Classifiers.angle_Classification(seq_list, x, label_list, labelFolds[i])
        # Save splitting angle, accuracy and runtime
        angles.append(splitAngle)
        runtimes.append(runtime)
        accuracies.append(accuracy)
    # Run the angle classifier on the test set with the mean splitting angle
    meanAngle = np.mean(np.array(angles))
    accuracy, angle, runtime = Classifiers.angle_Classification([], seqsTest, [], labelsTest, meanAngle)
    meanAccuracy = np.mean(np.array(accuracies))
    meanRunTime = np.mean(np.array(runtimes))
    print("Accuracy of test set of angle classifier using mean angle: ", accuracy)
    print("Mean Angle of angle classifier", meanAngle)
    print("Mean Runtime of angle classifier", meanRunTime)
    print("Mean Accuracy of angle classifier", meanAccuracy)
    print()
    
action_steps = []
Lengths = []
nrPerAction = []
for i, action in enumerate(action_names):
    # Select all sequences belonging to the current action.
    if (len(np.where(labelsTrain[:,0]==i+1)[0]) > 0):
        print('Now aligning: ',action_names[i][0][0])
        steps = []
        # Get the synthetic graph for the current action (Only run and walk)
        refSeq = DTWHelpers.getSyntheticGraph(i+1)
        # Create a new list of each sequence used to find the steps and align them
        for x in np.where(labelsTrain[:,0]==i+1)[0].tolist():
            steps.append(seqsTrain[x])
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

# Plot the movement of the z coordinate of the right foot
# fig = plt.figure()
# ax = plt.axes()
# for i in range(0, int(len(action_steps))):
#     for x in range(0,int(len(action_steps[i]))):
#         ax.plot(action_steps[i][x][11][:])


tensorList = []
for i in range(0,len(action_steps)):
    for x in range(0,len(action_steps[i])):
        temp = np.array(action_steps[i][x]).reshape([45,1,int(94)])
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
# Select cropping values on U2 and U3
crop_U2 = U2.shape[1]
crop_U3 = U3.shape[1]
# Crop the matrices and core tensor
U2 = U2[:, 0:crop_U2]
U3 = U3[:, 0:crop_U3]
core_S = core_S[:,0:crop_U2, 0:crop_U3]
Classifiers.U2_approximation([subSeqList[18]],np.array([labelsStacked[18]]),tensor, np.tensordot(core_S, U1, (0,1)), U2, U3, less_than=6e-13)
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

new_action_names = []
for i, action in enumerate(action_names):
    new_action_names.append(action_names[i][0][0])

# SVM classifier
if SVM_classifier:
    print()
    print("Starting SVM Classifier")
    print("Starting estimation of test set")
    # Pre-compute core_S x U1
    core_S_U1 = np.tensordot(core_S, U1, (0,1))
    U2_Estimate_list = []
    Estimates_Label_list = []

    start = datetime.now()
    # Compute U2 and label estimates
    U2_Estimates, Estimates_Labels = Classifiers.U2_approximation(seqsTest,labelsTest,tensor, core_S_U1, U2, U3, less_than=6e-13)
    U2_Estimate_list.append(U2_Estimates)
    Estimates_Label_list.append(Estimates_Labels)
    end = datetime.now()
    print("Runtime of estimation: ", end-start)

    U2_Estimate_list = np.array(U2_Estimate_list)
    # Iterate through estimates 
    for j, z in enumerate(U2_Estimate_list):
        newU2 = U2
        newLabels = labelsStacked
        # Create new U2 and Labels for plotting
        for i,x in enumerate(z):
            if Estimates_Label_list[j][i] == 5:
                newU2 = np.vstack((newU2, x.reshape(1,U2.shape[1])))
                newLabels = np.vstack((newLabels, [13]))
            elif Estimates_Label_list[j][i] == 9:
                newU2 = np.vstack((newU2, x.reshape(1,U2.shape[1])))
                newLabels = np.vstack((newLabels, [14]))

        # Plotting.plotU2(newU2, newLabels,np.append(np.append(new_action_names, 'run estimate'),'walk estimate'))
        # Plotting.plotU2(U2, labelsStacked,new_action_names)
        # Use below for showing errors in report
        # fig = plt.figure()
        # ax = plt.axes()
        # ax.plot(errors)
        # plt.show()

        print("Starting Training of SVM Model")
        
        # Run the SVM classifier on the estimated test set
        accuracy, runtime = Classifiers.SVM_Classification(U2, z, labelsStacked, Estimates_Label_list[j])
        print("Test accuracy: ", accuracy)
        print("Estimation + SVM runtime: ", runtime + end-start)
