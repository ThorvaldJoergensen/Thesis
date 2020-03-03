import numpy as np
from dtwalign import dtw as dtwalign
import math
import matplotlib.pyplot as plt


x = np.linspace(-np.pi,2.5*np.pi,100)
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
# for j in range(0,y2.shape[0]):
#     y22[j][0] = j
#     y22[j][1] = y2[j]-1
# Mindre sequence skal være query, altså først i metode kaldet
# res = dtwalign(y2, y1,step_pattern="asymmetric", open_begin=True)
# res.plot_path()
# print(res.path)

# y22path = res.path[:,0]
# y12path = res.path[:,1]
# # y12path = res.get_warping_path(target="reference")

# plt.plot(y2[y22path], c="g")
# plt.plot(y1[y12path],c="b")
# plt.show()

shift = 5
window_size = int(y1.shape[0]/5)
print(window_size)
threshold = 0.5
chuncklist = []
avg_y1 = np.average(y1)
print(avg_y1)
print(np.var(y1[15:70]))
while True:
    for i in range(0,y1.shape[0],shift):
        y1_window = y1[i:(i+window_size)]
        variance = np.var(y1_window)
        if variance > threshold:
            pass_counter = 0
            for x in y1_window:
                if x < avg_y1 and (pass_counter == 0 or pass_counter == 2):
                    pass_counter = pass_counter +1
                elif x > avg_y1 and pass_counter == 1:
                    pass_counter = pass_counter +1
            if pass_counter >= 3:
                chuncklist.append((i,y1_window))
    if len(chuncklist) >= 1:
        print(window_size)
        break
    if window_size >= y1.shape[0]:
        raise Exception("lorrrrrrt")
    window_size = window_size+shift

FirstofChunk = []
LastofChunk = []
for x in chuncklist:
    FirstofChunk.append(x[0])
    LastofChunk.append(x[0]+window_size)
    # print(x)
print(FirstofChunk)
print(LastofChunk)

fig = plt.figure()
ax = plt.axes()
ax.plot(y1,c="b", label="1")
ax.scatter(FirstofChunk,np.zeros([len(FirstofChunk)]), c="g")
ax.scatter(LastofChunk,np.zeros([len(LastofChunk)]), c="r")
# ax.plot(y22[:,0],y22[:,1],c="g", label="2")
plt.show()

