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
# currentlyTestingId = 50
# newSeq = subSeqList[currentlyTestingId]
# newSeqLabel = labelsStacked[currentlyTestingId]
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
# XXX Try different cropping values to see which is best
crop_U2 = 60
crop_U3 = 60
U1,U2,U3,core_S = TensorHelpers.svd(Tdiff)
# U2 = U2[:, 0:crop_U2]
# U3 = U3[:, 0:crop_U3]
# core_S = core_S[:,0:crop_U2, 0:crop_U3]
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

if SVM_classifier:
    print()
    print("Starting SVM Classifier")
    print("Starting Estimation of test set")
    core_S_U1 = np.tensordot(core_S, U1, (0,1))
    U2_Estimate_list = []
    Estimates_Label_list = []
    from datetime import datetime

    start = datetime.now()
    U2_Estimates, Estimates_Labels = Classifiers.U2_approximation(seqsTest,labelsTest,tensor, core_S_U1, U2, U3, less_than=4e-13)
    U2_Estimate_list.append(U2_Estimates)
    Estimates_Label_list.append(Estimates_Labels)
    end = datetime.now()
    print("Runtime of <4e-13 estimation: ", end-start)

    U2_Estimate_list = np.array(U2_Estimate_list)
    for j, z in enumerate(U2_Estimate_list):
        print("Running new version: ", j+1)
        newU2 = U2
        newLabels = labelsStacked
        for i,x in enumerate(z):
            if Estimates_Label_list[j][i] == 5:
                newU2 = np.vstack((newU2, x.reshape(1,U2.shape[1])))
                newLabels = np.vstack((newLabels, [13]))
            elif Estimates_Label_list[j][i] == 9:
                newU2 = np.vstack((newU2, x.reshape(1,U2.shape[1])))
                newLabels = np.vstack((newLabels, [14]))

            
        Plotting.plotU2(newU2, newLabels,np.append(np.append(action_names, 'RunEstimate'),'WalkEstimate'))
        # Use below for showing errors in report
        # fig = plt.figure()
        # ax = plt.axes()
        # ax.plot(errors)
        # fig2 = plt.figure()
        # ax2 = plt.axes()
        # ax2.plot(approximation_Errors)

        plt.show()
        #seqsTrain, seqsTest, labelsTrain, labelsTest = train_test_split(U2, labelsStacked, test_size = 0.20)
        print("Starting Training of SVM Model")

        seqFolds, labelFolds = Classifiers.create_Folds(U2,labelsStacked,5)
        accuracies = []
        runtimes = []
        # XXX Get the model from each training step and continue training it
        for i, x in enumerate(seqFolds):
            print("Training on fold: ", i)
            seq_list, label_list = Classifiers.getFoldSubList(seqFolds, labelFolds, i)
            accuracy, runtime = Classifiers.SVM_Classification_old(np.array(seq_list), np.array(x), np.array(label_list), np.array(labelFolds[i]))
            accuracies.append(accuracy)
            runtimes.append(runtime)
        #print(seqsTest.shape)
        accuracy, runtime = Classifiers.SVM_Classification_old(U2, z, labelsStacked, Estimates_Label_list[j])
        print("Test accuracy: ", accuracy)
        accuracies.append(accuracy)
        runtimes.append(runtime)
        print("Mean SVM accuracy: ", np.mean(np.array(accuracies)))
        print("Mean SVM runtime: ", np.mean(np.array(runtimes)))