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

    # Select a given subset of sequences
    if len(sys.argv) > 1:
        subSeqList, labelsStacked = TensorHelpers.getSequence(seqList, labels, sys.argv[1])
    else:
        subSeqList, labelsStacked = TensorHelpers.getSequence(seqList, labels)
    print("Result of get Sequence shape: ", len(subSeqList), len(subSeqList[1]))

    if len(subSeqList) == 0:
        print('Error in sequence result, please choose a valid sequence. [All, oneEach, allRunWalk, fiveRunWalk, allRun, allWalk, allMoving, fiveBalancedUneven]')
        exit()
    subSeqList = AlignData.spatial(subSeqList)
    subSeqList = DTWHelpers.reshapeTo45(subSeqList)
    action_steps = []
    Lengths = []
    nrPerAction = []
    for i, action in enumerate(action_names):

        if (len(np.where(labelsStacked[:,0]==i+1)[0]) > 0):
            print('Now aligning: ',action_names[i][0][0])
            steps = []
            refSeq = DTWHelpers.getSyntheticGraph(i+1)
            for x in np.where(labelsStacked[:,0]==i+1)[0].tolist():
                steps.append(subSeqList[x])
            steps, lengthList = DTWHelpers.multiDTW(steps,11, refSeq)
            Lengths.extend(lengthList)
            nrPerAction.append([i+1,len(lengthList)])
            action_steps.append(steps)
    Lengths = np.array(Lengths)
    medianLength = np.median(Lengths)-1
    fig = plt.figure()
    ax = plt.axes()
    for i in range(0, len(action_steps)):
        for x in range(0,len(action_steps[i])):
            ax.plot(action_steps[i][x][11][:])
    plt.show()
    tensorList = []
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

    data = np.concatenate((U2[36:72],U2[85:169]))
    # data = U2
    labels1 = np.array(labelList)
    labels = np.concatenate((labels1[36:72],labels1[85:169]))
    # labels = labels1
    classes = ['run', 'walk', 'boxing', 'golfswing', 'idle', 'jump', 'shoot', 'sit', 'sweepfloor', 'walkbalancing', 'walkuneventerrain', 'washwindow']

    labels2 = []
    for i, value in enumerate(labels):
        labels2.append(int(classes.index(value)))
    labels2 = np.array(labels2)
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels2, test_size = 0.25)

    # SVC Classifier
    # svclassifier = SVC(kernel='rbf')
    # svclassifier.fit(X_train, y_train)

    # y_pred = svclassifier.predict(X_test)

    # from sklearn.metrics import classification_report, confusion_matrix
    # print(classification_report(y_test,y_pred))

    # print(data.shape)
    # print(labels2.shape)
    # plt.scatter(data[:, 0], data[:, 1], c=labels2, s=50, cmap='autumn')


    # ax = plt.gca()
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()# create grid to evaluate model
    # xx = np.linspace(xlim[0], xlim[1], 30)
    # yy = np.linspace(ylim[0], ylim[1], 30)
    # YY, XX = np.meshgrid(yy, xx)
    # xy = np.vstack([XX.ravel(), YY.ravel()]).T
    # Z = svclassifier.decision_function(xy).reshape(XX.shape)# plot decision boundary and margins
    # ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
    #            linestyles=['--', '-', '--'])
    # # plot support vectors
    # ax.scatter(svclassifier.support_vectors_[:, 0], svclassifier.support_vectors_[:, 1], s=100,
    #            linewidth=1, facecolors='none', edgecolors='k')
    # plt.show()


    # SVC with GridSearchCV
    # Dimension of Train and Test set 
    print("Dimension of Train set",X_train.shape)
    print("Dimension of Test set",X_test.shape,"\n")

    # Transforming non numerical labels into numerical labels
    from sklearn import preprocessing
    encoder = preprocessing.LabelEncoder()

    # encoding train labels 
    encoder.fit(Y_train)
    Y_train = encoder.transform(Y_train)

    # encoding test labels 
    encoder.fit(Y_test)
    Y_test = encoder.transform(Y_test)

    #Total Number of Continous and Categorical features in the training set
    num_cols = pd.DataFrame(X_train)._get_numeric_data().columns
    print("Number of numeric features:",num_cols.size)
    #list(set(X_train.columns) - set(num_cols))


    names_of_predictors = list(pd.DataFrame(X_train).columns.values)

    # Scaling the Train and Test feature set 
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(pd.DataFrame(X_test))

    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.metrics import classification_report, confusion_matrix

    params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                        'C': [1, 10, 100, 1000, 10000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000, 10000]},
                        {'kernel': ['poly'], 'C': [1, 10, 100, 1000, 10000], 'degree' : [1, 2, 3, 5, 10], 'gamma': [1e-3, 1e-4]},
                        {'kernel': ['sigmoid'], 'C': [1, 10, 100, 1000, 10000], 'gamma': [1e-3, 1e-4]}]

    svm_model = GridSearchCV(SVC(), params_grid, cv=5)
    svm_model.fit(X_train_scaled, Y_train)

    print('Best score for training data:', svm_model.best_score_,"\n") 

    # View the best parameters for the model found using grid search
    print('Best C:',svm_model.best_estimator_.C,"\n") 
    print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
    print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")

    final_model = svm_model.best_estimator_
    Y_pred = final_model.predict(X_test_scaled)
    Y_pred_label = list(encoder.inverse_transform(Y_pred))

    print(confusion_matrix(Y_test,Y_pred_label))
    print("\n")
    print(classification_report(Y_test,Y_pred_label))

    print("Training set score for SVM: %f" % final_model.score(X_train_scaled , Y_train))
    print("Testing  set score for SVM: %f" % final_model.score(X_test_scaled  , Y_test ))

