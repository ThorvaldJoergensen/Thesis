import numpy as np
from dtwalign import dtw as dtwalign
import math
import matplotlib.pyplot as plt
from dtaidistance import dtw as dtw2
from dtaidistance import dtw_visualisation as dtwvis


x1 = range(0, 14)
y1 = np.array([1,1,2.5,2.5,3.5,4.5,3.5,3.5,4.5,3.5,3.5,3.5,5.5,4.5])
x2 = range(0, 14)
y2 = np.array([1,2,3,3,4,5,4,4,5,4,4,4,6,5])
from scipy.signal import savgol_filter
y1 = savgol_filter(y1, 13, 3)
y2 = savgol_filter(y2, 13, 3)


# x1 = range(0, 7)
# y1 = np.array([2,4,3,5,5,2,4])
# x2 = range(0, 9)
# y2 = np.array([2,4,3,5,2,5,5,2,4])



fig = plt.figure()
ax = plt.axes()
ax.plot(x1, y1,c="b")
ax.plot(x2, y2, c="r")

_,path = dtw2.warping_paths(y2, y1)
best_path = dtw2.best_path(path)
dtwvis.plot_warping(y2, y1, best_path)

# x = np.zeros([x1.shape[0],2])
# y12 = np.zeros([y1.shape[0],2])
# y22 = np.zeros([y2.shape[0],2])
# print(y1.shape)
# print(y2.shape)
# print(x2.shape)
# for i in range(0,x2.shape[0]):
#     x2[i][0] = i
#     x2[i][1] = x[i]
#     y12[i][0] = i
#     y12[i][1] = y1[i]
# for j in range(0,y2.shape[0]):
#     y22[j][0] = j
#     y22[j][1] = y2[j]-1
#Mindre sequence skal være query, altså først i metode kaldet
res = dtwalign(y1, y2,window_type="sakoechiba",window_size=3)
print(res.path)

y22path = res.path[:,0]
y12path = res.path[:,1]
# y12path = res.get_warping_path(target="reference")

fig = plt.figure()
ax = plt.axes()
y1 = savgol_filter(y1[y22path], 13, 3)
y2 = savgol_filter(y2[y12path], 13, 3)
ax.plot(y1,c="b")
ax.plot(y2, c="r")

res.plot_path()
plt.show()