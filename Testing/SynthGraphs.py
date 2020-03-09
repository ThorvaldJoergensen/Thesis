import numpy as np
import matplotlib.pyplot as plt

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

        print(fullStartToTop)
        print(fullTopToBottom)
        print(fullBottomToEnd)

        startSeq = np.linspace(start,top,fullStartToTop)
        topToBottomSeq = np.linspace(top,bottom,fullTopToBottom)
        bottomToEndSeq = np.linspace(bottom,end,fullBottomToEnd)
        fullSeq = []
        fullSeq.extend(startSeq)
        fullSeq.extend(topToBottomSeq)
        fullSeq.extend(bottomToEndSeq)
        print(fullSeq)
        fig = plt.figure()
        ax = plt.axes()
        from scipy.signal import savgol_filter
        yhat = savgol_filter(fullSeq, 51, 3)
        ax.plot(yhat)
        plt.show()
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

        print(fullStartToTop)
        print(fullTopToBottom)
        print(fullBottomToEnd)

        startSeq = np.linspace(start,top,fullStartToTop)
        topToBottomSeq = np.linspace(top,bottom,fullTopToBottom)
        bottomToEndSeq = np.linspace(bottom,end,fullBottomToEnd)
        fullSeq = []
        fullSeq.extend(startSeq)
        fullSeq.extend(topToBottomSeq)
        fullSeq.extend(bottomToEndSeq)
        print(fullSeq)
        fig = plt.figure()
        ax = plt.axes()
        from scipy.signal import savgol_filter
        yhat = savgol_filter(fullSeq, 51, 3)
        ax.plot(yhat)
        plt.show()
    return yhat