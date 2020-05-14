import numpy as np
import DTWHelpers
from dtwalign import dtw
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
import math
from datetime import datetime
import Plotting
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def alignment_Classification(seqsTrain, seqsTest, labelsTrain, labelsTest):
    #print("Starting alignment classifier")
    # Start timer and get synthetic graphs for each class
    start = datetime.now()
    runRef = DTWHelpers.getSyntheticGraph(5)
    walkRef = DTWHelpers.getSyntheticGraph(9)

    walkSeqs = []
    runSeqs = []
    # Split the set into running and walking sequences
    labelsTest = np.array(labelsTest)
    for x in np.where(labelsTest[:,0]==5)[0].tolist():
        runSeqs.append(seqsTest[x])
    for x in np.where(labelsTest[:,0]==9)[0].tolist():
        walkSeqs.append(seqsTest[x])
    # Align all sequences to both graphs
    run_runAligned, _ = DTWHelpers.multiDTW(runSeqs,11,runRef)
    run_walkAligned, _ = DTWHelpers.multiDTW(runSeqs,11,walkRef)

    walk_runAligned ,_ = DTWHelpers.multiDTW(walkSeqs,11,runRef)
    walk_walkAligned, _ = DTWHelpers.multiDTW(walkSeqs,11,walkRef)

    predictedLabels = []

    # Align all sequences to both synthetic graphs and then check their distance, lowest distance is then the guessed label
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
    
    # Check how many guesses were correct
    correct = 0
    for i in range(0,len(run_runAligned)):
        if predictedLabels[i] == 5:
            correct += 1
    
    for i in range(len(run_runAligned),len(run_runAligned)+len(walk_walkAligned)):
        if predictedLabels[i] == 9:
            correct += 1
    end = datetime.now()

    # print("Correct: ", correct)
    # print("Wrong: ", wrong)
    # print("Equal: ", equal)

    # fig = plt.figure()
    # ax = plt.axes()
    # # Plot the movement of the z coordinate of the right foot
    # for i in range(0, int(len(run_runAligned))):
    #     ax.plot(run_runAligned[i][11,:], c="b")
    # for i in range(0, int(len(walk_runAligned))):
    #     ax.plot(walk_runAligned[i][11,:], c="g")
    # ax.plot(runRef, c="r")

    # fig = plt.figure()
    # ax = plt.axes()
    # # Plot the movement of the z coordinate of the right foot
    # for i in range(0, int(len(run_walkAligned))):
    #     ax.plot(run_walkAligned[i][11,:], c="b")
    # for i in range(0, int(len(walk_walkAligned))):
    #     ax.plot(walk_walkAligned[i][11,:], c="g")
    # ax.plot(walkRef, c="r")
    # plt.show()

    runtime = end - start
    return correct/len(predictedLabels), runtime

# Function to claculate angle between two points (a,c) using a third point (b) as the meeting of a and c
def angle(a,b,c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # Some values were set to -1.000000002 and arccos only accepts value between -1.0 and 1.0
    if cosine_angle < -1.0:
        cosine_angle = -1.0
    Angle = np.arccos(cosine_angle)

    return np.degrees(Angle)

def angle_Classification(seqsTrain, seqsTest, labelsTrain, labelsTest, givenAngle=-1.0):
    #print("Starting angle classifier")
    start = datetime.now()
    # Check if we are given a trained angle split, if we are we can test right away
    if not givenAngle == -1:
        splitAngle = givenAngle
    else:
        maxAnglesTrain = []
        splitAngle = -1.0
        # For all sequences, get the angle for each frame between the upper and lower part of the right leg using the knee as middle point
        for j, x in enumerate(seqsTrain):
            angles = []
            seq = np.array(x)
            for i in range(0,seq.shape[1]):
                angles.append(angle(seq[9:12, i],seq[6:9, i], seq[3:6, i]))
            # Get the sharpest angle from the sequence
            maxAnglesTrain.append((labelsTrain[j],np.nanmin(np.array(angles))))
        maxAnglesTrain = np.array(maxAnglesTrain)
        run_max = -math.inf
        walk_min = math.inf
        
        # For every angle, check if it is running or walking and then compare to the current max or min to see if it needs to be replaced
        for i in maxAnglesTrain:
            if i[0] == [5]:
                if i[1] > run_max:
                    run_max = i[1]
            elif i[0] == [9]:
                if i[1] < walk_min:
                    walk_min = i[1]
        # Set the split to be halfway between the two outer values (running is lower than walking)
        splitAngle = run_max + ((walk_min - run_max) / 2)
    # print("Split angle calculated as: ", splitAngle)

    
    maxAnglesTest = []
    # Get max angle for every sequence in the test set
    for j, x in enumerate(seqsTest):
        angles = []
        seq = np.array(x)
        for i in range(0,seq.shape[1]):
            angles.append(angle(seq[9:12, i],seq[6:9, i], seq[3:6, i]))
        maxAnglesTest.append((labelsTest[j],np.nanmin(np.array(angles))))
    maxAnglesTest = np.array(maxAnglesTest)

    # Test using the calculated or given splitangle, if the max angle of the sequence is lower than split then it is a running sequence, else it is walking
    correct = 0
    for i in maxAnglesTest:
        if i[1] < splitAngle and i[0] == [5]:
            correct += 1
        elif i[1] > splitAngle and i[0] == [9]:
            correct += 1

    #print("Angle Classifier Accuracy: ", correct/maxAnglesTest.shape[0])
    end = datetime.now()
    #print("Runtime for angle classifier: ", end - start)
    runtime = end - start
    return correct/maxAnglesTest.shape[0], splitAngle, runtime

# Create a number of folds using a training set and the corresponding labels
def create_Folds(seqsTrain, labelsTrain, nrSplits):
    seqFolds = []
    labelFolds = []
    runSeqs = []
    runLabels = []
    # Split the running and walking sequences
    for x in np.where(labelsTrain[:,0]==5)[0].tolist():
        runSeqs.append(seqsTrain[x])
        runLabels.append([5])
    walkSeqs = []
    walkLabels = []
    for x in np.where(labelsTrain[:,0]==9)[0].tolist():
        walkSeqs.append(seqsTrain[x])
        walkLabels.append([9])
    # Calculate how many running and walking sequences each fold should contain
    nrRunPer = int(len(runSeqs)/nrSplits)
    nrWalkPer = int(len(walkSeqs)/nrSplits)
    # Create each fold using the above calculated values
    for i in range(0,nrSplits):
        seqFolds.append(runSeqs[nrRunPer*i:nrRunPer*(i+1)])
        seqFolds[i].extend(walkSeqs[nrWalkPer*i:nrWalkPer*(i+1)])
        labelFolds.append(runLabels[nrRunPer*i:nrRunPer*(i+1)])
        labelFolds[i].extend(walkLabels[nrWalkPer*i:nrWalkPer*(i+1)])
    return seqFolds, labelFolds

# Get a fold list not containing the one at the specified index
def getFoldSubList(foldList, labelList, index):
    seqsToTrain = []
    labelsToTrain = []
    seqsToTrain.extend(foldList[:index])
    seqsToTrain.extend(foldList[index+1:])
    labelsToTrain.extend(labelList[:index])
    labelsToTrain.extend(labelList[index+1:])
    seq_list = [item for sublist in seqsToTrain for item in sublist]
    label_list = [item for sublist in labelsToTrain for item in sublist]
    return seq_list, label_list

def SVM_Classification_old(seqsTrain, seqsTest, labelsTrain, labelsTest):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #print("Starting SVM classifier")
    start = datetime.now()
    
    batch_size = labelsTest.shape[0]
    x_data = tf.placeholder(shape=[None, seqsTrain.shape[1]], dtype=tf.float32)
    y_target = tf.placeholder(shape=[2, None], dtype=tf.float32)
    prediction_grid = tf.placeholder(shape=[None, seqsTrain.shape[1]], dtype=tf.float32)
    alpha = tf.Variable(tf.random_normal(shape=[2, batch_size]))

    y_vals1 = np.array([1 if y==5 else -1 for y in labelsTrain])
    y_vals2 = np.array([1 if y==9 else -1 for y in labelsTrain])
    labelsTrain = np.array([y_vals1, y_vals2])
    # labelsTrain = np.where(labelsTrain, 5, -1)
    # labelsTrain = np.where(labelsTrain, -1, 1)

    y_vals1 = np.array([1 if y==5 else -1 for y in labelsTest])
    y_vals2 = np.array([1 if y==9 else -1 for y in labelsTest])
    labelsTest = np.array([y_vals1, y_vals2])
    # labelsTest = np.where(labelsTest, 5, -1)
    # labelsTest = np.where(labelsTest, -1, 1)

    
    class1_idxs = np.flatnonzero(labelsTrain[0,:] == 1)
    class1_x = seqsTrain[class1_idxs]
    class1_r = labelsTrain[0, labelsTrain[0,:] == 1]
    class2_idxs = np.flatnonzero(labelsTrain[1,:] == 1)
    class2_x = seqsTrain[class2_idxs]
    class2_r = labelsTrain[1, labelsTrain[1,:] == 1]

    def reshape_matmul(mat, _size):
        v1 = tf.expand_dims(mat, 1)
        v2 = tf.reshape(v1, [2, _size, 1])
        return tf.matmul(v2, v1)

    # Gaussian (RBF) kernel
    gamma = tf.constant(-.01)
    dist = tf.reduce_sum(tf.square(x_data), 1)
    dist = tf.reshape(dist, [-1,1])
    sq_dist = tf.add(tf.subtract(dist, tf.multiply(2.0, tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
    kernel = tf.exp(tf.multiply(gamma, sq_dist))

    # Gaussian (RBF) prediction kernel
    rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1,1])
    rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1),[-1,1])
    pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
    pred_kernel = tf.exp(tf.multiply(gamma, pred_sq_dist))

    # Loss function
    first_term = tf.reduce_sum(alpha)
    b_vec_cross = tf.matmul(tf.transpose(alpha), alpha)
    y_target_cross = reshape_matmul(y_target, batch_size)
    second_term = tf.reduce_sum(tf.multiply(kernel, tf.multiply(b_vec_cross, y_target_cross)), [1, 2])
    loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))

    # Accuracy function
    pred_output = tf.matmul(tf.multiply(y_target, alpha), pred_kernel)
    prediction = tf.math.argmax(pred_output-tf.expand_dims(tf.reduce_mean(pred_output,1), 1), 0)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction,tf.argmax(y_target,0)), tf.float32))

    # Define optimizer and training step
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train_step = optimizer.minimize(loss)
    
    # Initialize variables
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Setup arrays of accuracy and loss values.
    loss_vec = []
    batch_accuracy = []
    test_accuracy = []
    test_loss = []
    np.random.seed(0) # set this for your experiments to compare the different kernels
    for i in range(10000):
        # Choose training points for current epoch
        batch_index = np.random.choice(seqsTrain.shape[0], size=batch_size)
        X = seqsTrain[batch_index]
        Y = labelsTrain[:, batch_index]

        # Run training step and get accuracy and loss values
        sess.run(train_step, feed_dict={x_data: X, y_target: Y})
        temp_loss = sess.run(loss, feed_dict={x_data: X, y_target: Y})
        loss_vec.append(temp_loss)
        acc_temp = sess.run(accuracy, feed_dict={x_data: X, y_target: Y, prediction_grid: X})
        batch_accuracy.append(acc_temp)

    # After training, run on test set.
    temp_test_accuracy = sess.run(accuracy, feed_dict={x_data: seqsTest, y_target: labelsTest, prediction_grid: seqsTest})
    #print ("SVM classifier accuracy: ", temp_test_accuracy)

    end = datetime.now()
    #print("Runtime for SVM classifier: ", end - start)
    runtime = end - start

    # fig = plt.figure(figsize=(13,4))

    # # plot batch accuracy
    # ax1 = fig.add_subplot(121)
    # ax1.plot(batch_accuracy, color="black", linewidth=0.1)
    # ax1.set_title("Accuracy per batch")
    # ax1.set_xlabel("Batch")
    # ax1.set_ylabel("Accuracy")

    # # plot loss over time
    # ax2 = fig.add_subplot(122)
    # ax2.plot(loss_vec, color="black", linewidth=0.1)
    # ax2.set_title("Loss per batch")
    # ax2.set_xlabel("Batch")
    # ax2.set_ylabel("Loss")

    # plt.show()
    return temp_test_accuracy, runtime


def SVM_Classification(X_train, X_test, Y_train, Y_test):
    start = datetime.now()

    # Transforming non numerical labels into numerical labels
    from sklearn import preprocessing
    encoder = preprocessing.LabelEncoder()

    # encoding train labels 
    encoder.fit(Y_train.ravel())
    Y_train = encoder.transform(Y_train.ravel())

    # encoding test labels 
    encoder.fit(Y_test.ravel())
    Y_test = encoder.transform(Y_test.ravel())

    # Scaling the Train and Test feature set 
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(pd.DataFrame(X_test))

    from sklearn.model_selection import GridSearchCV

    params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                        'C': [1, 10, 100, 1000, 10000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000, 10000]},
                        {'kernel': ['poly'], 'C': [1, 10, 100, 1000, 10000], 'degree' : [1, 2, 3, 5, 10], 'gamma': [1e-3, 1e-4]},
                        {'kernel': ['sigmoid'], 'C': [1, 10, 100, 1000, 10000], 'gamma': [1e-3, 1e-4]}]
  
    svm_model = GridSearchCV(SVC(), params_grid, cv=5, iid=False)
    svm_model.fit(X_train_scaled, Y_train.ravel())

    # View the best parameters for the model found using grid search
    # print('Best C:',svm_model.best_estimator_.C,"\n") 
    # print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
    # print('Best degree:',svm_model.best_estimator_.degree,"\n")
    # print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")

    final_model = svm_model.best_estimator_
    Y_pred = final_model.predict(X_test_scaled)
    Y_pred_label = list(encoder.inverse_transform(Y_pred))

    # print(confusion_matrix(Y_test,Y_pred_label))
    # print("\n")
    # print(classification_report(Y_test,Y_pred_label))

    # print("Training set score for SVM: %f" % final_model.score(X_train_scaled , Y_train))
    # print("Testing  set score for SVM: %f" % final_model.score(X_test_scaled  , Y_test ))
    end = datetime.now()
    return final_model.score(X_test_scaled, Y_test), (end-start)

def U2_approximation(seq_list, label_list,tensor, core_S_U1, U2, U3, less_than=-1, change=-1, it = 50):
    WalkTest = []
    RunTest = []
    # print("Parameters: ", less_than, change, it)
    # Split into running and walking
    for x in np.where(label_list[:,0]==5)[0].tolist():
        RunTest.append(seq_list[x])

    for x in np.where(label_list[:,0]==9)[0].tolist():
        WalkTest.append(seq_list[x])

    def Approximation(seq_list,tensor,core_S_U1,U2,U3, less_than, change, it):    
        # Perform step selection and alignment to the middle synthetic graph for all given sequences
        stepSeqs, _ = DTWHelpers.multiDTW(seq_list, 11, DTWHelpers.getSyntheticGraph(0), test=False) 
        mean_new_shape = np.mean(tensor, axis=(2,1))
        # Compute the mean body and mean sequence
        mean_new_shape.reshape(45,1)
        mean_body = np.zeros((45,tensor.shape[2]))
        from datetime import datetime
        for i, x in enumerate(mean_new_shape):
            mean_body[i] = np.resize(x, tensor.shape[2])

        f_hatU2 = lambda u2,U3 : np.add(np.tensordot(np.tensordot(core_S_U1, u2, (0,0)), U3, (0,1)), mean_body) #Gives matrix of size 45x94
        f_hatU3 = lambda u2,u3 : np.add(np.tensordot(np.tensordot(core_S_U1, u3, (1,0)), u2, (0,0)), mean_new_shape) #Gives vector of size U1.shape[0]
        
        # Helper function to calculate the approximation error of a given sequence compared to the original sequence
        def approximation_Error(f_hat, f_true):
            sum_k = 0
            for i,x in enumerate(f_hat):
                sum_k += np.linalg.norm(x - f_true[i])
            return (1/94)*sum_k
        
        U2_Estimates = []
        Labels = []

        for g, seq in enumerate(stepSeqs):
            # Calculate a randow U2 row, within the bounds of the real U2 matrix
            blank_U2 = np.zeros(U2.shape[1])
            for i, x in enumerate(blank_U2):
                blank_U2[i] = np.random.uniform(np.min(U2[:,i]), high=np.max(U2[:,i]))
            u2_hat = blank_U2#np.random.uniform(np.min(U2), high=np.max(U2), size=U2.shape[0])#rand_U2
            # Compute a random U3 Matrix, within the bounds of the real U3 matrix
            blank_U3 = np.zeros(U3.shape)
            for i, x in enumerate(blank_U3):
                blank_U3[i,:] = np.random.uniform(np.min(U3[i,:]), high=np.max(U3[i,:]), size=(U3.shape[1]))
            U3_hat = blank_U3#np.random.uniform(np.min(U3), high=np.max(U3), size=(U3.shape[0],U3.shape[1]))#rand_U3
            U2_list = []
            errors = []
            iteration = 0
            
            approximation_Errors = []
            prev_U2 = u2_hat
            prev_U3 = U3_hat
            start_U2 = u2_hat
            start_U3 = U3_hat
            last_approx = approximation_Error(f_hatU2(u2_hat,U3_hat),seq)
            for j in range (0,it): # Run for the spefcified number of iterations
                if not less_than == -1: # If Less_than has been set, then stop approximating the current sequence once the approximation error goes below less_than
                    if approximation_Error(f_hatU2(u2_hat,U3_hat),seq) <= less_than:
                        break
                    if approximation_Error(f_hatU2(u2_hat,U3_hat),seq) - last_approx > -(0.5*less_than) and approximation_Error(f_hatU2(u2_hat,U3_hat),seq) - last_approx < (0.5*less_than) and j > 0: # If the approximation error has not changed mroe than half of less_than since last iteration and we are not in the first iteration, then break
                        break
                if not change == -1: # If change is set, then the approximation stops when the approximation error changes more in a single iteration than the change parameter
                    if approximation_Error(f_hatU2(u2_hat,U3_hat),seq) - last_approx > change:
                        u2_hat = prev_U2
                        U3_hat = prev_U3
                        break
                iteration += 1
                start = datetime.now()
                
                # Estimate a u3 matrix using the current estimated u2 row (Eq. XXX)
                M3 = np.tensordot(core_S_U1, u2_hat, (0,0))
                M3pinv = np.linalg.pinv(M3)
                new_U3_list = np.zeros(U3_hat.shape)
                for i, x in enumerate(U3_hat):
                    new_U3_list[i] = np.matmul(np.subtract(seq[:,i], mean_new_shape), M3pinv)
                U3_hat = new_U3_list
                    
                # Create the M2 stack as used in Eq. XXX
                M2List = []
                for i, x in enumerate(U3_hat):
                    M2List.append(np.transpose(np.tensordot(core_S_U1,x,(1,0))))
                    
                M2 = None
                for i, x in enumerate(M2List):
                    if M2 is None:
                        M2 = x
                    else:
                        M2 = np.vstack((M2, x))
                f_hatList = []
                # Create the F_hat stack as used in Eq. XXX
                for i, x in enumerate(U3_hat):
                    f_hatList.append((np.subtract(seq[:,i], mean_new_shape)).reshape(45,1))
                f_hat_matrix = None
                for i, x in enumerate(f_hatList):
                    if f_hat_matrix is None:
                        f_hat_matrix = x
                    else:
                        f_hat_matrix = np.vstack((f_hat_matrix, x))
                # Estimate u2 using the above stacks as performed in Eq. XXX
                u2_hat = np.matmul(np.linalg.pinv(M2), f_hat_matrix).reshape(U2.shape[1])

                end = datetime.now()
                errors.append(0.5 * np.abs(np.linalg.norm(f_hatU2(u2_hat,U3_hat) - seq))**2)
                U2_list.append(u2_hat)
                appr_error = approximation_Error(f_hatU2(u2_hat,U3_hat),seq)

                approximation_Errors.append(appr_error)
                prev_U2 = u2_hat
                prev_U3 = U3_hat
                
            # if iteration > 1:
            #     fig = plt.figure()
            #     ax = plt.axes()
            #     ax.set_xlabel('Iterations')
            #     ax.set_ylabel('Error')
            #     fig.suptitle("Approximation error")
            #     ax.plot(approximation_Errors)
            #     plt.show()
            # print("Final Approximation error: ",appr_error)
            U2_Estimates.append(u2_hat)
            # Use the below code to animate the estimated sequence
            if g == -1:
                newMatrix2 = np.tensordot(core_S_U1, u2_hat, (0,0))
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
                # fig = plt.figure()
                # ax = plt.axes()
                # ax.plot(seq[11,:])
                # plt.show()

                # Testing
                newMatrix2 = np.tensordot(core_S_U1, u2_hat, (0,0))
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
                ax = fig.add_subplot(2,1,1, projection='3d')
                ax.set_title('Reconstructed vs original sequence')
                ax.set_ylabel('Reconstructed')
                sct, = ax.plot([], [], [], "o", markersize=2)

                from matplotlib import cm
                jet = cm.get_cmap('jet',FirstFrameModel.shape[1]) 
                distance = 3
                for i in range(0, FirstFrameModel.shape[1]):
                    if i % 10 == 0:
                        ax.scatter(xs[i*15:i*15+14], [y + i * distance for y in ys[i*15:i*15+14]], zs[i*15:i*15+14], color=jet(i))          
                        # Right leg
                        ax.plot([xs[i*15+0], xs[i*15+1]], [ys[i*15+0] + i * distance , ys[i*15+1] + i * distance ], [zs[i*15+0], zs[i*15+1]], color=jet(i), markersize=2)
                        ax.plot([xs[i*15+1], xs[i*15+2]], [ys[i*15+1] + i * distance , ys[i*15+2] + i * distance ], [zs[i*15+1], zs[i*15+2]], color=jet(i), markersize=2)
                        ax.plot([xs[i*15+2], xs[i*15+3]], [ys[i*15+2] + i * distance , ys[i*15+3] + i * distance ], [zs[i*15+2], zs[i*15+3]], color=jet(i), markersize=2)
                        # Left leg
                        ax.plot([xs[i*15+0], xs[i*15+4]], [ys[i*15+0] + i * distance , ys[i*15+4] + i * distance ], [zs[i*15+0], zs[i*15+4]], color=jet(i), markersize=2)
                        ax.plot([xs[i*15+4], xs[i*15+5]], [ys[i*15+4] + i * distance , ys[i*15+5] + i * distance ], [zs[i*15+4], zs[i*15+5]], color=jet(i), markersize=2)
                        ax.plot([xs[i*15+5], xs[i*15+6]], [ys[i*15+5] + i * distance , ys[i*15+6] + i * distance ], [zs[i*15+5], zs[i*15+6]], color=jet(i), markersize=2)
                        # Spine
                        ax.plot([xs[i*15+0], xs[i*15+7]], [ys[i*15+0] + i * distance , ys[i*15+7] + i * distance ], [zs[i*15+0], zs[i*15+7]], color=jet(i), markersize=2)
                        ax.plot([xs[i*15+7], xs[i*15+8]], [ys[i*15+7] + i * distance , ys[i*15+8] + i * distance ], [zs[i*15+7], zs[i*15+8]], color=jet(i), markersize=2)
                        # Right arm
                        ax.plot([xs[i*15+7], xs[i*15+9]], [ys[i*15+7] + i * distance , ys[i*15+9] + i * distance ], [zs[i*15+7], zs[i*15+9]], color=jet(i), markersize=2)
                        ax.plot([xs[i*15+9], xs[i*15+10]], [ys[i*15+9] + i * distance , ys[i*15+10] + i * distance ], [zs[i*15+9], zs[i*15+10]], color=jet(i), markersize=2)
                        ax.plot([xs[i*15+10], xs[i*15+11]], [ys[i*15+10] + i * distance , ys[i*15+11] + i * distance ], [zs[i*15+10], zs[i*15+11]], color=jet(i), markersize=2)
                        # Left arm
                        ax.plot([xs[i*15+7], xs[i*15+12]], [ys[i*15+7] + i * distance , ys[i*15+12] + i * distance ], [zs[i*15+7], zs[i*15+12]], color=jet(i), markersize=2)
                        ax.plot([xs[i*15+12], xs[i*15+13]], [ys[i*15+12] + i * distance , ys[i*15+13] + i * distance ], [zs[i*15+12], zs[i*15+13]], color=jet(i), markersize=2)
                        ax.plot([xs[i*15+13], xs[i*15+14]], [ys[i*15+13] + i * distance , ys[i*15+14] + i * distance ], [zs[i*15+13], zs[i*15+14]], color=jet(i), markersize=2)

                # Limit coordinates for all axes
                ax.set_xlim(40,-40)
                ax.set_ylim(275,-20)
                ax.set_zlim(-30,30)

                # Set labels
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")

                ax.grid(False)
                plt.axis('off')

                FirstFrameModel = stepSeqs[0]

                xs = []
                ys = []
                zs = []

                # Split the data into x,y,z coordinates for each frame
                for frm in range(0, FirstFrameModel.shape[1]):
                    for j in range(0, 45, 3):
                        xs.append(FirstFrameModel[j][frm])
                        ys.append(FirstFrameModel[j+1][frm])
                        zs.append(FirstFrameModel[j+2][frm])

                ax = fig.add_subplot(2,1,2, projection='3d')
                ax.set_ylabel('Original')
                sct, = ax.plot([], [], [], "o", markersize=2)

                from matplotlib import cm
                jet = cm.get_cmap('jet',FirstFrameModel.shape[1]) 
                for i in range(0, FirstFrameModel.shape[1]):
                    if i % 10 == 0:
                        ax.scatter(xs[i*15:i*15+14], [y + i * distance for y in ys[i*15:i*15+14]], zs[i*15:i*15+14], color=jet(i))          
                        # Right leg
                        ax.plot([xs[i*15+0], xs[i*15+1]], [ys[i*15+0] + i * distance , ys[i*15+1] + i * distance ], [zs[i*15+0], zs[i*15+1]], color=jet(i), markersize=2)
                        ax.plot([xs[i*15+1], xs[i*15+2]], [ys[i*15+1] + i * distance , ys[i*15+2] + i * distance ], [zs[i*15+1], zs[i*15+2]], color=jet(i), markersize=2)
                        ax.plot([xs[i*15+2], xs[i*15+3]], [ys[i*15+2] + i * distance , ys[i*15+3] + i * distance ], [zs[i*15+2], zs[i*15+3]], color=jet(i), markersize=2)
                        # Left leg
                        ax.plot([xs[i*15+0], xs[i*15+4]], [ys[i*15+0] + i * distance , ys[i*15+4] + i * distance ], [zs[i*15+0], zs[i*15+4]], color=jet(i), markersize=2)
                        ax.plot([xs[i*15+4], xs[i*15+5]], [ys[i*15+4] + i * distance , ys[i*15+5] + i * distance ], [zs[i*15+4], zs[i*15+5]], color=jet(i), markersize=2)
                        ax.plot([xs[i*15+5], xs[i*15+6]], [ys[i*15+5] + i * distance , ys[i*15+6] + i * distance ], [zs[i*15+5], zs[i*15+6]], color=jet(i), markersize=2)
                        # Spine
                        ax.plot([xs[i*15+0], xs[i*15+7]], [ys[i*15+0] + i * distance , ys[i*15+7] + i * distance ], [zs[i*15+0], zs[i*15+7]], color=jet(i), markersize=2)
                        ax.plot([xs[i*15+7], xs[i*15+8]], [ys[i*15+7] + i * distance , ys[i*15+8] + i * distance ], [zs[i*15+7], zs[i*15+8]], color=jet(i), markersize=2)
                        # Right arm
                        ax.plot([xs[i*15+7], xs[i*15+9]], [ys[i*15+7] + i * distance , ys[i*15+9] + i * distance ], [zs[i*15+7], zs[i*15+9]], color=jet(i), markersize=2)
                        ax.plot([xs[i*15+9], xs[i*15+10]], [ys[i*15+9] + i * distance , ys[i*15+10] + i * distance ], [zs[i*15+9], zs[i*15+10]], color=jet(i), markersize=2)
                        ax.plot([xs[i*15+10], xs[i*15+11]], [ys[i*15+10] + i * distance , ys[i*15+11] + i * distance ], [zs[i*15+10], zs[i*15+11]], color=jet(i), markersize=2)
                        # Left arm
                        ax.plot([xs[i*15+7], xs[i*15+12]], [ys[i*15+7] + i * distance , ys[i*15+12] + i * distance ], [zs[i*15+7], zs[i*15+12]], color=jet(i), markersize=2)
                        ax.plot([xs[i*15+12], xs[i*15+13]], [ys[i*15+12] + i * distance , ys[i*15+13] + i * distance ], [zs[i*15+12], zs[i*15+13]], color=jet(i), markersize=2)
                        ax.plot([xs[i*15+13], xs[i*15+14]], [ys[i*15+13] + i * distance , ys[i*15+14] + i * distance ], [zs[i*15+13], zs[i*15+14]], color=jet(i), markersize=2)

                # Limit coordinates for all axes
                ax.set_xlim(40,-40)
                ax.set_ylim(275,-20)
                ax.set_zlim(-30,30)

                # Set labels
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")

                ax.grid(False)
                plt.axis('off')

                plt.show()

        return U2_Estimates

    # Approximate running and walking sequences
    Run_Estimates = []
    if len(RunTest) > 0:
        # print("Estimating Running steps ", len(RunTest))
        Run_Estimates = Approximation(RunTest,tensor,core_S_U1,U2,U3, less_than, change, it)
    Walk_Estimates = []
    if len(WalkTest) > 0:
        # print("Estimating Walking Steps ", len(WalkTest))
        Walk_Estimates = Approximation(WalkTest,tensor,core_S_U1,U2,U3, less_than, change, it)
    # Create estimate and label lists
    estimates  = None
    labels = None
    for x in Run_Estimates:
        if estimates is None:
           estimates = x
           labels = [5]
        else:
            estimates = np.vstack((estimates,x))
            labels = np.vstack((labels,[5]))
    
    for x in Walk_Estimates:
        if estimates is None:
           estimates = x
           labels = [9]
        else:
            estimates = np.vstack((estimates,x))
            labels = np.vstack((labels,[9]))
    
    return estimates,labels