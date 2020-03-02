import numpy as np
from dtwalign import dtw as dtwalign
import math
import matplotlib.pyplot as plt


x = np.linspace(-np.pi,0,50)
y1 = np.sin(x)
xtest = np.linspace(-np.pi,0,25)
y2 = np.sin(xtest)

x2 = np.zeros([x.shape[0],2])
y12 = np.zeros([y1.shape[0],2])
y22 = np.zeros([y2.shape[0],2])
print(y1.shape)
print(y2.shape)
print(x.shape)
for i in range(0,x2.shape[0]):
    x2[i][0] = i
    x2[i][1] = x[i]
    y12[i][0] = i
    y12[i][1] = y1[i]
for j in range(0,y2.shape[0]):
    y22[j][0] = j
    y22[j][1] = y2[j]-1
# Mindre sequence skal være query, altså først i metode kaldet
res = dtwalign(y2, y1,step_pattern="asymmetric", open_begin=True)
res.plot_path()
print(res.path)

y22path = res.path[:,0]
y12path = res.path[:,1]
# y12path = res.get_warping_path(target="reference")

plt.plot(y2[y22path], c="g")
plt.plot(y1[y12path],c="b")
plt.show()

fig = plt.figure()
ax = plt.axes()
ax.plot(y12[:,0], y12[:,1],c="b", label="1")
ax.plot(y22[:,0],y22[:,1],c="g", label="2")
plt.show()

