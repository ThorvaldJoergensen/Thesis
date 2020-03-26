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
    
    newSeq = subSeqList.pop(0)
    newSeqLabel = labelsStacked[0]
    labelsStacked = np.delete(labelsStacked,0).reshape([119,1])

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
    for i, x in enumerate(mean_new_shape):
        mean_body[i] = np.resize(x, tensor.shape[2])

    f_hat = lambda u2,u3 : np.add(np.tensordot(np.tensordot(np.tensordot(core_S, U1, (0,1)), u2, (0,0)), u3, (0,0)), mean_new_shape)
    init_U2 = np.mean(U2, axis=0)
    init_U3 = np.mean(U3, axis=0)
    U3_hat = np.zeros(U3.shape)
    for x in stepSeqs:
        for i, frame in enumerate(x):
            opt_fun = lambda u3: 0.5 * np.abs(np.linalg.norm(f_hat(init_U2,u3) - frame[i]))**2
            # print(opt_fun(U3[i,:]))
            U3_hat[i] = opti.minimize(opt_fun, U3[i,:], method='Newton-CG', jac=rosen_der, options={'maxiter':10}).x

    # Continue from here:
    #   Currently need to stack following results (Eq. 22)
    #   Repeat untill convergence
    f_hat = lambda u2,U3 : np.add(np.tensordot(np.tensordot(np.tensordot(core_S, U1, (0,1)), u2, (0,0)), U3, (0,1)), mean_body)
    opt_fun = lambda u2: 0.5 * np.abs(np.linalg.norm(f_hat(u2,U3_hat) - frame[i]))**2
    u2_hat = opti.minimize(opt_fun, init_U2, method='Newton-CG', jac=rosen_der, options={'maxiter':10}).x
    
    print(U2.shape)
    print(np.array(u2_hat).shape)
    Plotting.plotU2(np.vstack((U2, u2_hat)), np.vstack((labelsStacked, [13])), np.append(action_names, 'Test'))
    print(newSeqLabel)
    plt.show()
    print(init_U2.shape)
    print(init_U3.shape)
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