from scipy.spatial import procrustes
import numpy as np
import matplotlib.pyplot as plt

a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')
b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')
mtx1, mtx2, disparity = procrustes(a, b)
print(disparity)

plt.plot(a[:,0], a[:,1], label="A")
plt.plot(b[:,0], b[:,1], label="B")


plt.plot(mtx1[:,0], mtx1[:,1], label="mtx1")
plt.plot(mtx2[:,0], mtx2[:,1], label="mtx2")

# plt.plot((mtx2*np.transpose(b))[:,0], (mtx2*np.transpose(b))[:,1], label="mtx2*b")

plt.legend()
plt.show()