import numpy as np
import DTWHelpers
from dtwalign import dtw
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
import math

def alignment_Classification(runSeqs, walkSeqs):
    runRef = DTWHelpers.getSyntheticGraph(5)
    walkRef = DTWHelpers.getSyntheticGraph(9)

    run_runAligned, _ = DTWHelpers.multiDTW(runSeqs,11,runRef)
    run_walkAligned, _ = DTWHelpers.multiDTW(runSeqs,11,walkRef)

    walk_runAligned ,_ = DTWHelpers.multiDTW(walkSeqs,11,runRef)
    walk_walkAligned, _ = DTWHelpers.multiDTW(walkSeqs,11,walkRef)

    predictedLabels = []

    for i in range(0,len(run_runAligned)):
        res = dtw(run_runAligned[i][11,:],runRef,dist_only=True,step_pattern="typeIVc")
        runDist = res.distance
        res = dtw(run_walkAligned[i][11,:],walkRef,dist_only=True,step_pattern="typeIVc")
        walkDist = res.distance
        if runDist < walkDist:
            predictedLabels.append(5)
        elif walkDist < runDist:
            predictedLabels.append(9)
        elif walkDist == runDist:
            predictedLabels.append(-1)

    for i in range(0,len(walk_walkAligned)):
        res = dtw(walk_runAligned[i][11,:],runRef,dist_only=True,step_pattern="typeIVc")
        runDist = res.distance
        res = dtw(walk_walkAligned[i][11,:],walkRef,dist_only=True,step_pattern="typeIVc")
        walkDist = res.distance
        if runDist < walkDist:
            predictedLabels.append(5)
        elif walkDist < runDist:
            predictedLabels.append(9)
        elif walkDist == runDist:
            predictedLabels.append(-1)
        
    correct = 0
    wrong = 0
    equal = 0
    for i in range(0,len(run_runAligned)):
        if predictedLabels[i] == 5:
            correct += 1
        elif predictedLabels[i] == -1:
            equal += 1
        elif predictedLabels[i] == 9:
            wrong += 1
    
    for i in range(len(run_runAligned),len(run_runAligned)+len(walk_walkAligned)):
        if predictedLabels[i] == 9:
            correct += 1
        elif predictedLabels[i] == -1:
            equal += 1
        elif predictedLabels[i] == 5:
            wrong += 1

    # print("Correct: ", correct)
    # print("Wrong: ", wrong)
    # print("Equal: ", equal)
    print("Alignment Classifier Accuracy: ", correct/len(predictedLabels))

    fig = plt.figure()
    ax = plt.axes()
    # Plot the movement of the z coordinate of the right foot
    for i in range(0, int(len(run_runAligned))):
        ax.plot(run_runAligned[i][11,:], c="b")
    for i in range(0, int(len(walk_runAligned))):
        ax.plot(walk_runAligned[i][11,:], c="g")
    ax.plot(runRef, c="r")

    fig = plt.figure()
    ax = plt.axes()
    # Plot the movement of the z coordinate of the right foot
    for i in range(0, int(len(run_walkAligned))):
        ax.plot(run_walkAligned[i][11,:], c="b")
    for i in range(0, int(len(walk_walkAligned))):
        ax.plot(walk_walkAligned[i][11,:], c="g")
    ax.plot(walkRef, c="r")
    plt.show()

def angle(a,b,c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # Some values were set to -1.000000002 and arccos only accepts value between -1.0 and 1.0
    if cosine_angle < -1.0:
        cosine_angle = -1.0
    Angle = np.arccos(cosine_angle)

    return np.degrees(Angle)

def angle_Classification(seqs, labels):
    maxAngles = []
    for j, x in enumerate(seqs):
        angles = []
        seq = np.array(x)
        for i in range(0,seq.shape[1]):
            angles.append(angle(seq[9:12, i],seq[6:9, i], seq[3:6, i]))
        maxAngles.append((labels[j],np.nanmin(np.array(angles))))
    maxAngles = np.array(maxAngles)
    run_max = 0
    walk_min = math.inf
    correct = 0
    wrong = 0
    for i in maxAngles:
        if i[1] < 97.5 and i[0] == 5:
            correct += 1
        elif i[1] > 97.5 and i[0] == 9:
            correct += 1
        else:
            wrong += 1

    # print("Correct: ", correct)
    # print("Wrong: ", wrong)
    print("Angle Classifier Accuracy: ", correct/maxAngles.shape[0])
    # for i in maxAngles:
    #     if i[0] == 5:
    #         if i[1] > run_max:
    #             run_max = i[1]
    #     elif i[0] == 9:
    #         if i[1] < walk_min:
    #             walk_min = i[1]
    # print("Max run angle: ", run_max)
    # print("Min walk angle: ", walk_min)

        
    # plot1[j][1] = angle(AlignedSeqs[i, 9:12, j],AlignedSeqs[i, 6:9, j], AlignedSeqs[i, 3:6, j]) #Leg

def SVM_Classification(data, labels):
    # data = np.concatenate((U2[36:72],U2[85:169]))
    classes = ['run', 'walk', 'boxing', 'golfswing', 'idle', 'jump', 'shoot', 'sit', 'sweepfloor', 'walkbalancing', 'walkuneventerrain', 'washwindow']

    # labels2 = []
    # for i, value in enumerate(labels):
    #     labels2.append(int(classes.index(value)))
    labels2 = labels#np.array(labels2)
    labels2 = np.where(labels2==5, 0, labels2)
    labels2 = np.where(labels2==9, 1, labels2)
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels2, test_size = 0.25)

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

