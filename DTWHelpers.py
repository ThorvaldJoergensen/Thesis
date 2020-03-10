import numpy as np
import glob, os, math
import matplotlib.pyplot as plt
from dtaidistance import dtw as dtw2
from dtwalign import dtw as dtwalign

#Create Synthethic graphs for DTW Alignement
def getSyntheticGraph(id):
    if id == 5:
        start = -33.0
        top = -20.0
        bottom = -36.0
        end = -34.0

        fullLength = 95

        baseLength = 78.0
        onePercent = baseLength/100
        startToTop = 28.0
        topToBottom = 34.0
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
        
        # fig = plt.figure()
        # ax = plt.axes()
        from scipy.signal import savgol_filter
        yhat = savgol_filter(fullSeq, 51, 3)
        # ax.plot(yhat)
        # plt.show()
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
        # fig = plt.figure()
        # ax = plt.axes()
        from scipy.signal import savgol_filter
        yhat = savgol_filter(fullSeq, 51, 3)
        # ax.plot(yhat)
        # plt.show()
    return yhat

def findSteps(seq):
    secondApproach = False
    shift = int(seq.shape[0]/20)
    window_size = int(seq.shape[0]/10)
    threshold = np.var(seq)
    chuncklist = []
    avg_seq = np.average(seq)
    while True:
        for i in range(0,seq.shape[0],shift):
            seq_window = seq[i:(i+window_size)]
            variance = np.var(seq_window)
            if variance >= threshold*0.9:
                pass_counter = 0
                for x in seq_window:
                    if (not secondApproach and ((x < avg_seq and pass_counter == 0) or (x < avg_seq and pass_counter == 2))) or (secondApproach and ((x > avg_seq and pass_counter == 0) or (x > avg_seq and pass_counter == 2))):
                        pass_counter += 1
                    if (not secondApproach and (x > avg_seq and pass_counter == 1)) or (secondApproach and (x < avg_seq and pass_counter == 1)):
                        pass_counter += 1
                if pass_counter >= 3:
                    chuncklist.append((i,seq_window))
        if len(chuncklist) >= 2:
            break
        if window_size >= seq.shape[0]:
            if secondApproach and len(chuncklist) == 0:
                raise RuntimeError("No steps found")
            secondApproach = True
            window_size = int(seq.shape[0]/10)
        window_size = window_size+shift

    FirstofChunk = []
    LastofChunk = []
    for x in chuncklist:
        FirstofChunk.append(x[0])
        LastofChunk.append(x[0]+window_size)
    FirstofChunk.sort()
    LastofChunk.sort()
    FinalFirst = []
    FinalLast = []

    for i, val in enumerate(FirstofChunk):
        if i != 0 and FirstofChunk[i-1] >= val-shift:
            continue
        else:
            FinalFirst.append(val)

    for i, val in enumerate(LastofChunk):
        if i < len(LastofChunk)-1 and LastofChunk[i+1] <= val+shift:
            continue
        else:
            FinalLast.append(val)

    for i, val in enumerate(FinalFirst):
        for j in range(0,i):
            if FinalLast[j] > val and val > FinalFirst[i-1]+((FinalLast[j] - FinalFirst[i-1])/2):
                FinalFirst[i] = FinalLast[j]
            elif val < FinalFirst[i-1]+((FinalLast[i-1] - FinalFirst[i-1])/2):
                FinalFirst.pop(i)
                FinalLast.pop(i)
    return FinalFirst, FinalLast

def multiDTW(seqs, id, refSeq):
    aligned = []
    for i in range(0, len(seqs)):
        aligned.append(seqs[i][id][:])
    
    stepSeqs = []
    for i, x in enumerate(aligned):
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
    
    longestId = -1
    maxLength = -1
    lengthList = []
    for i, x in enumerate(stepSeqs):
        lengthList.append(len(x[1][0]))
        if len(x[1][0]) > maxLength:
            maxLength = len(x[1][0])
            longestId = i
    print("LongestId: ", longestId)
    print("MaxLength: ", maxLength)
    for i, x in enumerate(stepSeqs):
        res = dtwalign(x[1][id,:], refSeq,step_pattern="typeIVc")
        stepSeqs[i][1] = stepSeqs[i][1][:,res.get_warping_path(target="query")]
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

def smoothSeq(seq, path):
    counter = 1
    for i in range(0, len(path), counter):
        counter = 0
        for j in range(i+1, len(path)):
            if path[i] == path[j]:
                counter += 1
            else:
                break
        
        combinationValue = 1.0
        for k in range(1, counter+1):
            combinationValue -= 1/(counter+1)
            if i+counter+1 < len(path) - 1:
                try:
                    seq[:, i+k] = combinationValue*seq[:, i]+(1-combinationValue)*seq[:, i+counter]
                except:
                    print(i)
                    print(k)
                    print(counter)
                    print(len(path))
                    raise ValueError("THIS FUCKED UP")
            else: 
                if i == len(path) - 1:
                    break
                distanceToPrev = seq[:, i] - seq[:, i-1]
                seq[:, i+k] = combinationValue*seq[:, i]+(1-combinationValue)*(distanceToPrev+seq[:, i])
            # if k < seq.shape[0]-4:
            #     seq[i] = combinationValue*seq[i]+(1-combinationValue)*seq[i+counter]
            # else:
            #     distanceToPrevX = seq[j] - seq[j-3]
            #     distanceToPrevY = seq[j+1] - seq[j-2]
            #     distanceToPrevZ = seq[j+2] - seq[j-1]
            #     tempList.append(combinationValue*seq[j]+(1-combinationValue)*(distanceToPrevX+seq[j]))
            #     tempList.append(combinationValue*seq[j+1]+(1-combinationValue)*(distanceToPrevY+seq[j+1]))
            #     tempList.append(combinationValue*seq[j+2]+(1-combinationValue)*(distanceToPrevZ+seq[j+2]))
            # if counter-k >= 1:
            #     combinationValue -= 1/(counter+1)
            # else:
            #     combinationValue -= 1/counter
        if counter == 0:
            counter = 1
    return seq

