import numpy as np
import DTWHelpers
from dtwalign import dtw
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
import math
from datetime import datetime
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def alignment_Classification(seqsTrain, seqsTest, labelsTrain, labelsTest):
    print("Starting alignment classifier")
    start = datetime.now()
    runRef = DTWHelpers.getSyntheticGraph(5)
    walkRef = DTWHelpers.getSyntheticGraph(9)
    
    walkSeqs = []
    runSeqs = []

    for x in np.where(labelsTest[:,0]==5)[0].tolist():
        runSeqs.append(seqsTest[x])
    for x in np.where(labelsTest[:,0]==9)[0].tolist():
        walkSeqs.append(seqsTest[x])

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

    print("Alignment Classifier Accuracy: ", correct/len(predictedLabels))
    print("Runtime for alignment classifier: ", end - start)
    return correct/len(predictedLabels)

def angle(a,b,c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # Some values were set to -1.000000002 and arccos only accepts value between -1.0 and 1.0
    if cosine_angle < -1.0:
        cosine_angle = -1.0
    Angle = np.arccos(cosine_angle)

    return np.degrees(Angle)

def angle_Classification(seqsTrain, seqsTest, labelsTrain, labelsTest):
    print("Starting angle classifier")
    start = datetime.now()
    maxAnglesTrain = []
    splitAngle = -1.0
    for j, x in enumerate(seqsTrain):
        angles = []
        seq = np.array(x)
        for i in range(0,seq.shape[1]):
            angles.append(angle(seq[9:12, i],seq[6:9, i], seq[3:6, i]))
        maxAnglesTrain.append((labelsTrain[j],np.nanmin(np.array(angles))))
    maxAnglesTrain = np.array(maxAnglesTrain)
    run_max = -math.inf
    walk_min = math.inf
    
    for i in maxAnglesTrain:
        if i[0] == 5:
            if i[1] > run_max:
                run_max = i[1]
        elif i[0] == 9:
            if i[1] < walk_min:
                walk_min = i[1]
    splitAngle = run_max + ((walk_min - run_max) / 2)
    print("Split angle calculated as: ", splitAngle)

    
    maxAnglesTest = []
    for j, x in enumerate(seqsTest):
        angles = []
        seq = np.array(x)
        for i in range(0,seq.shape[1]):
            angles.append(angle(seq[9:12, i],seq[6:9, i], seq[3:6, i]))
        maxAnglesTest.append((labelsTest[j],np.nanmin(np.array(angles))))
    maxAnglesTest = np.array(maxAnglesTest)

    correct = 0
    for i in maxAnglesTest:
        if i[1] < splitAngle and i[0] == 5:
            correct += 1
        elif i[1] > splitAngle and i[0] == 9:
            correct += 1

    # print("Correct: ", correct)
    # print("Wrong: ", wrong)
    print("Angle Classifier Accuracy: ", correct/maxAnglesTest.shape[0])
    end = datetime.now()
    print("Runtime for angle classifier: ", end - start)
    return correct/maxAnglesTest.shape[0]
    # print("Max run angle: ", run_max)
    # print("Min walk angle: ", walk_min)

        
    # plot1[j][1] = angle(AlignedSeqs[i, 9:12, j],AlignedSeqs[i, 6:9, j], AlignedSeqs[i, 3:6, j]) #Leg

def SVM_Classification(seqsTrain, seqsTest, labelsTrain, labelsTest):
    print("Starting SVM classifier")
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
    # alpha_sum = tf.reduce_sum(alpha)
    # r_matrix = tf.matmul(y_target, tf.transpose(y_target))
    # alpha_prod = tf.matmul(tf.transpose(alpha), alpha)
    # double_sum = tf.reduce_sum(tf.multiply(kernel, tf.multiply(alpha_prod, r_matrix)))
    # loss = tf.negative(tf.subtract(alpha_sum, double_sum))

    first_term = tf.reduce_sum(alpha)
    b_vec_cross = tf.matmul(tf.transpose(alpha), alpha)
    y_target_cross = reshape_matmul(y_target, batch_size)
    second_term = tf.reduce_sum(tf.multiply(kernel, tf.multiply(b_vec_cross, y_target_cross)), [1, 2])
    loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))

    # Accuracy function
    # prediction_output = tf.multiply(pred_kernel, tf.multiply(y_target,alpha))
    # prediction = tf.arg_max(prediction_output-tf.expand_dims(tf.reduce_mean(prediction_output,1), 1), 0)
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction,tf.argmax(y_target,0)), tf.float32))

    pred_output = tf.matmul(tf.multiply(y_target, alpha), pred_kernel)
    prediction = tf.arg_max(pred_output-tf.expand_dims(tf.reduce_mean(pred_output,1), 1), 0)
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

        # For every 1000 epochs, print loss and accuracy
        if (i+1)%1000==0:
            print("Step #{}".format(i+1))
            print("Loss = ", temp_loss)
            print("Accuracy = ", acc_temp)
    # After training, run on test set.
    temp_test_accuracy = sess.run(accuracy, feed_dict={x_data: seqsTest, y_target: labelsTest, prediction_grid: seqsTest})
    print ("SVM classifier accuracy: ", temp_test_accuracy)

    end = datetime.now()
    print("Runtime for SVM classifier: ", end - start)

    fig = plt.figure(figsize=(13,4))

    # plot batch accuracy
    ax1 = fig.add_subplot(121)
    ax1.plot(batch_accuracy, color="black", linewidth=0.1)
    ax1.set_title("Accuracy per batch")
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Accuracy")

    # plot loss over time
    ax2 = fig.add_subplot(122)
    ax2.plot(loss_vec, color="black", linewidth=0.1)
    ax2.set_title("Loss per batch")
    ax2.set_xlabel("Batch")
    ax2.set_ylabel("Loss")

    # find boundaries for contour plot
    abscissa_min, abscissa_max = seqsTest[:, 0].min()-1, seqsTest[:, 0].max()+1
    ordinate_min, ordinate_max = seqsTest[:, 1].min()-1, seqsTest[:, 1].max()+1

    # generate mesh grid of points
    xx, yy = np.meshgrid(
        np.linspace(abscissa_min, abscissa_max, 1000),
        np.linspace(ordinate_min, ordinate_max, 1000)
    )
    grid_points = np.squeeze(np.c_[yy.ravel(), xx.ravel()])
    print(grid_points.shape)

    # find predictions for grid of points
    grid_preds = sess.run(prediction, feed_dict={
        x_data: seqsTest,
        y_target: labelsTest,
        prediction_grid: grid_points
    })
    grid_preds = grid_preds.reshape(xx.shape)

    # plot our decision boundary
    plt.imshow(
        grid_preds,
        extent=[abscissa_min, abscissa_max, ordinate_min, ordinate_max],
        origin="lower",
        cmap="bwr",
        aspect="auto",
        alpha=0.375
    )
    plt.contour(xx, yy, grid_preds, 1, colors="black", alpha=0.5)

    # plot our points
    plt.scatter(class1_x[:, 0], class1_x[:, 1],
        label = "Class 1 (+1)",
        color = "none",
        edgecolor = "red"
    )
    plt.scatter(class2_x[:, 0], class2_x[:, 1],
        label = "Class 2 (-1)",
        color = "none",
        edgecolor = "blue"
    )

    # add title and legend
    plt.title("Decision boundary of trained SVM")
    plt.legend(loc="upper left", framealpha=0.25)

    plt.show()


def SVM_Classification_old(data, labels):
    print("Starting SVM classifier")
    start = datetime.now()
    # data = np.concatenate((U2[36:72],U2[85:169]))

    # labels2 = []
    # for i, value in enumerate(labels):
    #     labels2.append(int(classes.index(value)))
    labels2 = labels#np.array(labels2)
    labels2 = np.where(labels2==5, 0, labels2)
    labels2 = np.where(labels2==9, 1, labels2)
    labels2 = np.ravel(labels2)
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels2, test_size = 0.25)

    # SVC with GridSearchCV
    # Dimension of Train and Test set 
    # print("Dimension of Train set",X_train.shape)
    # print("Dimension of Test set",X_test.shape,"\n")

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
    # print("Number of numeric features:",num_cols.size)
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

    # print('Best score for training data:', svm_model.best_score_,"\n") 

    # View the best parameters for the model found using grid search
    # print('Best C:',svm_model.best_estimator_.C,"\n") 
    # print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
    # print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")

    final_model = svm_model.best_estimator_
    Y_pred = final_model.predict(X_test_scaled)
    Y_pred_label = list(encoder.inverse_transform(Y_pred))

    # print(confusion_matrix(Y_test,Y_pred_label))
    # print("\n")
    # print(classification_report(Y_test,Y_pred_label))

    print("Training set score for SVM: %f" % final_model.score(X_train_scaled , Y_train))
    print("Testing  set score for SVM: %f" % final_model.score(X_test_scaled  , Y_test ))
    end = datetime.now()
    print("Runtime for SVM classifier: ", end - start)

