import numpy as np
import matplotlib.pyplot as plt

keypoints = np.load('D:/Programming/Programs_Py/est3d/vid2.npy')

pt0 = keypoints[:, 1, :]
pt1 = keypoints[:, 4, :]

diff = pt0 - pt1
dist = np.linalg.norm(diff, axis=1)

plt.plot(range(len(dist)), dist)
plt.show()

