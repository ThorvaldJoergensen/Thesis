from dtw import *
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

from fastdtw import fastdtw
from dtaidistance import dtw as dtw2
from dtaidistance import dtw_visualisation as dtwvis


#DtaiDistance testing:
s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0, 0])
path = dtw2.warping_path(s1, s2)
dtwvis.plot_warping(s1, s2, path, filename="warp.png")

#Fastdtw testing:
x = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]])
y = np.array([[2,2], [3,3], [4,4]])
distance, path = fastdtw(x, y, dist=euclidean)
print("distance: ",distance)
print(path)

fig = plt.figure()
ax = plt.axes()
ax.scatter(x[:,0], x[:,1],s=15,c="b")
ax.scatter(y[:,0],y[:,1],s=15,c="g")
plt.show()

#dtw-python Testing:
idx = np.linspace(0,6.28,num=100)
query = np.sin(idx) + np.random.uniform(size=100)/10.0

## A cosine is for template; sin and cos are offset by 25 samples
template = np.cos(idx)

## Find the best match with the canonical recursion formula
alignment = dtw(query, template, keep_internals=True)

## Display the warping curve, i.e. the alignment curve
alignment.plot(type="threeway")

## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
dtw(query, template, keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(6, "c"))\
    .plot(type="twoway",offset=-2)

## See the recursion relation, as formula and diagram
print(rabinerJuangStepPattern(6,"c"))
rabinerJuangStepPattern(6,"c").plot()

seq1 = np.zeros(10)
seq2 = np.zeros(10)

for i, x in enumerate(seq1):
    seq1[i] = i*3

for i, x in enumerate(seq2):
    if i+1 < seq1.shape[0]:
        seq2[i] = seq1[i+1]
    else:
        seq2[i] = (i+1)*3

alignment2 = dtw(seq2, seq1, keep_internals=True)

alignment2.plot(type="threeway")

dtw(seq2, seq1, keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(6, "c"))\
    .plot(type="twoway",offset=-2)