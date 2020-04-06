import numpy as np
import glob, os, math
import matplotlib.pyplot as plt
from dtaidistance import dtw as dtw2
from dtwalign import dtw as dtwalign

#Create Synthethic graphs for DTW Alignement
def getSyntheticGraph(id):
    # If the id is running sequence
    if id == 5:
        # Setup parameters for where the graph changes direction and for how long it should be
        start = -33.0
        top = -20.0
        bottom = -36.0
        end = -34.0

        fullLength = 95

        # Observed lengths between the setup points
        baseLength = 78.0
        onePercent = baseLength/100
        startToTop = 28.0
        topToBottom = 34.0
        bottomToEnd = 16.0

        # How much of a percentage of the graph is between each point
        percent1 = startToTop/onePercent
        percent2 = topToBottom/onePercent
        percent3 = bottomToEnd/onePercent

        # Calculate how many points need to be between the setup points, based on the observed percentages
        onePercentOfFull = fullLength/100
        fullStartToTop = int(onePercentOfFull * percent1)
        fullTopToBottom = int(onePercentOfFull * percent2)
        fullBottomToEnd = int(onePercentOfFull * percent3)

        # Create the points to insert between the setup points
        startSeq = np.linspace(start,top,fullStartToTop)
        topToBottomSeq = np.linspace(top,bottom,fullTopToBottom)
        bottomToEndSeq = np.linspace(bottom,end,fullBottomToEnd)
        fullSeq = []
        fullSeq.extend(startSeq)
        fullSeq.extend(topToBottomSeq)
        fullSeq.extend(bottomToEndSeq)
        
        # Smooth the calculated graph to remove the completely straight lines
        from scipy.signal import savgol_filter
        yhat = savgol_filter(fullSeq, 51, 3)
    #If the id is walking sequence
    elif id == 9:
        start = -34.0
        top = -26.0
        bottom = -34.0
        end = -33.0

        fullLength = 95

        baseLength = 95.0
        onePercent = baseLength/100
        startToTop = 49.0
        topToBottom = 30.0
        bottomToEnd = 16.0

        percent1 = startToTop/onePercent
        percent2 = topToBottom/onePercent
        percent3 = bottomToEnd/onePercent

        onePercentOfFull = fullLength/100
        fullStartToTop = int(onePercentOfFull * percent1)
        fullTopToBottom = int(onePercentOfFull * percent2)
        fullBottomToEnd = int(onePercentOfFull * percent3)

        startSeq = np.linspace(start,top,fullStartToTop)
        topToBottomSeq = np.linspace(top,bottom,fullTopToBottom)
        bottomToEndSeq = np.linspace(bottom,end,fullBottomToEnd)
        fullSeq = []
        fullSeq.extend(startSeq)
        fullSeq.extend(topToBottomSeq)
        fullSeq.extend(bottomToEndSeq)
        from scipy.signal import savgol_filter
        yhat = savgol_filter(fullSeq, 51, 3)
    #This is the adjoint sequence
    elif id == 0:
        start = -33.5
        top = -23.0
        bottom = -35.0
        end = -33.5

        fullLength = 95

        baseLength = 86.5
        onePercent = baseLength/100
        startToTop = 38.5
        topToBottom = 32.0
        bottomToEnd = 16.0

        percent1 = startToTop/onePercent
        percent2 = topToBottom/onePercent
        percent3 = bottomToEnd/onePercent

        onePercentOfFull = fullLength/100
        fullStartToTop = int(onePercentOfFull * percent1)
        fullTopToBottom = int(onePercentOfFull * percent2)
        fullBottomToEnd = int(onePercentOfFull * percent3)

        startSeq = np.linspace(start,top,fullStartToTop)
        topToBottomSeq = np.linspace(top,bottom,fullTopToBottom)
        bottomToEndSeq = np.linspace(bottom,end,fullBottomToEnd)
        fullSeq = []
        fullSeq.extend(startSeq)
        fullSeq.extend(topToBottomSeq)
        fullSeq.extend(bottomToEndSeq)
        from scipy.signal import savgol_filter
        yhat = savgol_filter(fullSeq, 51, 3)
    return yhat

# Method to find the steps in a given sequence
def findSteps(seq):
    secondApproach = False
    # Setup how much we shift the window, how large the window is and what the threshold of the steps should be
    shift = int(seq.shape[0]/20)
    window_size = int(seq.shape[0]/10)
    threshold = np.var(seq)
    chuncklist = []
    avg_seq = np.average(seq)
    while True:
        # Look through the sequence, moving the window shift spaces each iteration
        for i in range(0,seq.shape[0],shift):
            seq_window = seq[i:(i+window_size)]
            variance = np.var(seq_window)
            if variance >= threshold*0.9:
                pass_counter = 0
                # If the window contains something we think might be interesting to check, we then check if it passes the average of the sequence twice (starting below, going above, then below)
                # If secondApproach is true then we check if the window passes the average twice in the opposite order (starting above average, going below and then above)
                # This is to catch steps in sequences that start or end differently from the majority
                for x in seq_window:
                    if (not secondApproach and ((x < avg_seq and pass_counter == 0) or (x < avg_seq and pass_counter == 2))) or (secondApproach and ((x > avg_seq and pass_counter == 0) or (x > avg_seq and pass_counter == 2))):
                        pass_counter += 1
                    if (not secondApproach and (x > avg_seq and pass_counter == 1)) or (secondApproach and (x < avg_seq and pass_counter == 1)):
                        pass_counter += 1
                # If we pas the average twice, append the window to the chuncklist
                if pass_counter >= 3:
                    chuncklist.append((i,seq_window))
        # If we have found 2 or more steps (or the same step twice) we continue
        if len(chuncklist) >= 2:
            break
        # If the window i larger than the sequence, we try the secondapproach. If we already tried the second approach then we have not found any steps, and something is wrong so we raise an exception
        if window_size >= seq.shape[0]:
            if secondApproach and len(chuncklist) == 0:
                raise RuntimeError("No steps found")
            secondApproach = True
            window_size = int(seq.shape[0]/10)
        # Increase window size
        window_size = window_size+shift

    FirstofChunk = []
    LastofChunk = []
    # Add all windows begin and end values to new lists
    for x in chuncklist:
        FirstofChunk.append(x[0])
        LastofChunk.append(x[0]+window_size)
    # These lists may be out of order, so we sort them
    FirstofChunk.sort()
    LastofChunk.sort()
    FinalFirst = []
    FinalLast = []

    # Check if the previous saved window starts within the shift value of this windows start, if so skip the current point
    for i, val in enumerate(FirstofChunk):
        if i != 0 and FirstofChunk[i-1] >= val-shift:
            continue
        else:
            FinalFirst.append(val)

    # Check if the next saved window ends within the shift value of this windows end, if so skip the current point
    for i, val in enumerate(LastofChunk):
        if i < len(LastofChunk)-1 and LastofChunk[i+1] <= val+shift:
            continue
        else:
            FinalLast.append(val)
    # If the beginning of a window (x) lies inside of another window (y), first check if x lies closer to the end of y than the beginning, if it does move x to the end of y.
    # If not, then remove x and the end of y, but keep the beginning of y and x's end point
    for i, val in enumerate(FinalFirst):
        for j in range(0,i):
            if FinalLast[j] > val and val > FinalFirst[i-1]+((FinalLast[j] - FinalFirst[i-1])/2):
                FinalFirst[i] = FinalLast[j]
            elif val < FinalFirst[i-1]+((FinalLast[i-1] - FinalFirst[i-1])/2):
                FinalFirst.pop(i)
                FinalLast.pop(i)
    return FinalFirst, FinalLast

# Align the given sequence list to the reference sequence using the values in the place given from id
def multiDTW(seqs, id, refSeq, test=True):
    aligned = []
    # Append the single coordinate values of each sequence
    for i in range(0, len(seqs)):
        aligned.append(seqs[i][id][:])
    
    stepSeqs = []
    for i, x in enumerate(aligned):
        # Find all steps in each sequence and put them in a new list
        finalFirst, finalLast = findSteps(np.array(aligned[i]))
        for j, x in enumerate(finalFirst):
            temp = np.array(seqs[i][:])
            stepSeqs.append([i, temp[:,x:finalLast[j]]])
            # fig = plt.figure()
            # ax = plt.axes()
            # ax.plot(temp[11,:],c="b", label="1")
            # ax.scatter(finalFirst,np.full([len(finalFirst)],-33), c="g")
            # ax.scatter(finalLast,np.full([len(finalLast)],-33), c="r")
            # plt.show()
    
    adjoint_reference = getSyntheticGraph(0)
    longestId = -1
    maxLength = -1
    lengthList = []
    # Find the lengths of all steps and the longest length
    for i, x in enumerate(stepSeqs):
        lengthList.append(len(x[1][0]))
        if len(x[1][0]) > maxLength:
            maxLength = len(x[1][0])
            longestId = i
    # print("LongestId: ", longestId)
    # print("MaxLength: ", maxLength)
    # # Align each step to the refernce sequence given and save the new sequence
    for i, x in enumerate(stepSeqs):
        res = dtwalign(x[1][id,:], refSeq,step_pattern="typeIVc")
        # if i == 4:
        #     smoothSeq(stepSeqs[i][1][:,res.get_warping_path(target="query")], res.get_warping_path(target="query"), True)
        stepSeqs[i][1] = smoothSeq(stepSeqs[i][1][:,res.get_warping_path(target="query")], res.get_warping_path(target="query"))
        # Align to adjoint reference graphs
        if test:
            res = dtwalign(stepSeqs[i][1][id,:], adjoint_reference,step_pattern="typeIVc")
            stepSeqs[i][1] = smoothSeq(stepSeqs[i][1][:,res.get_warping_path(target="query")], res.get_warping_path(target="query"))
    return np.array(stepSeqs)[:,1], lengthList

def multiDTW_new_old(seqs, id):
    aligned = []
    for i in range(0, seqs.shape[1]):
        aligned.append(seqs[id, i, :])
    aligned = np.array(aligned)
    #print(dtw2.distance_matrix_fast(aligned, parallel=True))
    sim_matrix = np.zeros([seqs.shape[1],seqs.shape[1]])
    iPos = 0
    JPos = 0
    firstI = 0
    firstJ = 0
    minDist = math.inf
    firstPath = []
    for i, x in enumerate(aligned):
        for j, y in enumerate(aligned):
            if i == j:
                sim_matrix[i][j] = math.inf
            else:
                distance = dtwalign(y,x,window_type="typeIds",window_size=int(min(aligned[JPos].shape[0], x.shape[0])/10), dist_only=True).distance
                sim_matrix[i][j] = distance
                if distance < minDist:
                    minDist = distance
                    firstI = i
                    firstJ = j
                    iPos = i
                    JPos = j
    AlignedSeqs = np.zeros([45,aligned.shape[0],aligned[iPos].shape[0]])
    AlignedSeqs[:, iPos,:] = seqs[:,iPos]
    AlignedIds = [iPos]
    aligned = np.array(aligned)
    for g in range(0,aligned.shape[0]-1):
        # newISeqs, newJSeqs = scaleSeqs(aligned[JPos], AlignedSeqs[iPos, id, :])
        # _,paths = dtw2.warping_paths(aligned[JPos], AlignedSeqs[iPos, id, :], window=int(min(aligned[JPos].shape[0], AlignedSeqs[iPos,id,:].shape[0])/10))
        # res = dtwalign(newJSeqs, newISeqs, dist='matching')
        res = dtwalign(aligned[JPos], AlignedSeqs[id, iPos, :],step_pattern="typeIds", window_size=int(min(aligned[JPos].shape[0], x.shape[0])/10))
        path = res.get_warping_path(target="query")
        # path = np.array(dtw2.best_path(paths))
        JAligned = np.zeros([45,AlignedSeqs[:,iPos].shape[1]])
        JAligned = seqs[:, JPos][:, path]
        # for j in range(0,AlignedSeqs[iPos].shape[1]):
        #     JAligned[:,j] = seqs[JPos][:,int(path[j][0])]
        AlignedSeqs[:,JPos,:] = JAligned
        AlignedIds.append(JPos)
        if g != aligned.shape[0]-2:
            newSim = np.zeros([len(AlignedIds),aligned.shape[0]])
            for f,x in enumerate(AlignedIds):
                newSim[f] = sim_matrix[x]
            minDist = math.inf
            for i,x in enumerate(newSim):
                for j,d in enumerate(newSim[i]):
                    if j in AlignedIds or i == j:
                        continue
                    else:
                        if newSim[i][j] < minDist:
                            minDist = newSim[i][j]
                            iPos = AlignedIds[i]
                            JPos = j

    return AlignedSeqs

def multiDTW_old(seqs, id):
    aligned = []
    for i in range(0,seqs.shape[1]):
        aligned.append(seqs[id,i,:])
    aligned = np.array(aligned)
    #print(dtw2.distance_matrix_fast(aligned, parallel=True))
    sim_matrix = np.zeros([seqs.shape[1],seqs.shape[1]])
    iPos = 0
    JPos = 0
    firstI = 0
    firstJ = 0
    minDist = math.inf
    firstPath = []
    for i, x in enumerate(aligned):
        for j, y in enumerate(aligned):
            if i == j:
                sim_matrix[i][j] = math.inf
            else:
                distance = dtw2.distance_fast(aligned[j], x)
                sim_matrix[i][j] = distance
                if distance < minDist:
                    minDist = distance
                    firstI = i
                    firstJ = j
                    iPos = i
                    JPos = j

    # print("Id's: ", iPos, JPos)
    # print(sim_matrix)
    AlignedSeqs = np.zeros([45,aligned.shape[0],aligned[iPos].shape[0]])
    AlignedSeqs[:,iPos,:] = seqs[:,iPos,:]
    AlignedIds = {iPos}
    aligned = np.array(aligned)
    for g in range(0,aligned.shape[0]-1):
        _,paths = dtw2.warping_paths(aligned[JPos], AlignedSeqs[id,iPos,:])
        path = np.array(dtw2.best_path(paths))
        JAligned = np.zeros([45,AlignedSeqs[:,iPos,:].shape[1]])
        for j in range(0,AlignedSeqs[:,iPos,:].shape[1]):
            JAligned[:,j] = seqs[:,JPos,:][:,int(path[j][0])]
        AlignedSeqs[:,JPos,:] = JAligned
        AlignedIds.add(JPos)
        if g != aligned.shape[0]-2:
            newSim = np.zeros([len(AlignedIds),aligned.shape[0]])
            for f,x in enumerate(AlignedIds):
                newSim[f] = sim_matrix[x]
            minDist = math.inf
            for i,x in enumerate(newSim):
                for j,d in enumerate(newSim[i]):
                    if j in AlignedIds or i == j:
                        continue
                    else:
                        if newSim[i][j] < minDist:
                            minDist = newSim[i][j]
                            iPos = i
                            JPos = j
    return AlignedSeqs

#Reshape given Sequence list from number of frames * 3, 15 to 45, number of frames
def reshapeTo45(Aligned):
    RightFormFull = []
    for i, x in enumerate(Aligned):
        W1copy = np.zeros([45, int(x.shape[0]/3)])
        for j in range(0,int(x.shape[0]/3)):
            k = 3*j
            shape = x[k:k+3,:]
            points = np.zeros([45])
            for l in range(0,15):
                points[l*3] = shape[0][l]
                points[(l*3)+2] = shape[1][l]
                points[(l*3)+1] = shape[2][l]
            W1copy[:,j] = points
        RightFormFull.append(W1copy)
    return RightFormFull

# Smoothes a given sequence using the path provided
def smoothSeq(seq, path, debug = False):
    # Setup the counter to use while looking through the path
    counter = 1
    if debug:
        print("path", path)
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(seq[11,:])
    for i in range(0, len(path), counter):
        counter = 0
        # Check the next spots if they are set to the same value as the current one, if they are add 1 to the counter.
        for j in range(i+1, len(path)):
            # Maybe check the if statement if it is correct (print path[i] and path[j], see whats up and see if counter is counted up)
            if path[i] == path[j]:
                counter += 1
            # Check if there is a dip (backwards alignment) and repeat frames in this case before smoothing
            elif path[i] > path[j]:
                seq[:, j] = seq[:, i]
                path[j] = path[i]
                counter += 1
            else:
                break
        # Setup the value that is used to determine how large each part of the linear combination should be
        combinationValue = 1.0
        for k in range(1, counter+1):
            combinationValue -= 1/(counter+1)
            if i+counter+1 < len(path) - 1:
                try:
                    seq[:, i+k] = combinationValue*seq[:, i]+(1-combinationValue)*seq[:, i+counter+1]
                except:
                    raise ValueError("Something went wrong in smoothing, please try again")
            else: 
                if i == len(path) - 1:
                    break
                distanceToPrev = seq[:, i] - seq[:, i-1]
                seq[:, i+k] = combinationValue*seq[:, i]+(1-combinationValue)*(distanceToPrev+seq[:, i])
        if counter == 0:
            counter = 1
    return seq

