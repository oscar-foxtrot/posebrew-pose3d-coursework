import os
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture



#keypoints_3d = temp_seq/file_107.npy
#keypoints_3d = np.load("predictions/file_102_file_107_fused_file_101_fused.npy", allow_pickle=True)
#keypoints_3d = np.load("temp_seq/file_101_file_102_alternated.npy", allow_pickle=True)
#keypoints_3d = np.load("temp_seq/file_101_file_102_alternated.npy", allow_pickle=True)

offset_1 = 0
#keypoints_3d = np.load("predictions/file_101_file_102_fused.npy", allow_pickle=True)[230 + offset_1:550 + offset_1]
keypoints_3d = np.load("temp_seq/file_101_file_102_alternated.npy", allow_pickle=True)
#keypoints_3d = np.load('predictions/file_102_aligned.npy', allow_pickle=True)[230 + offset_1:550 + offset_1]
keypoints_3d = keypoints_3d.astype(float) 
print(len(keypoints_3d))


h36m_pts = [(3,2), (2,1), (1, 0), (0, 4), (4, 5), (5, 6), \
    (13, 12), (12, 11), (11, 8), (8, 14), (14, 15), (15, 16), \
    (8, 9), (9, 10), (8, 7), (7, 0)]


upper = keypoints_3d[:, 2] - keypoints_3d[:, 3]  # Right
lower = keypoints_3d[:, 5] - keypoints_3d[:, 6]  # Left

norm_upper = np.linalg.norm(upper, axis=1)
norm_lower = np.linalg.norm(lower, axis=1)

plt.plot(norm_upper, label='Триангуляция', color='black')
#plt.plot(norm_lower, label='Триангуляция', color='black')
#plt.title('wdw')
plt.xlabel('Момент времени (кадр)')
plt.ylabel('Длина, в условных единицах')




#keypoints_3d = temp_seq/file_107.npy
#keypoints_3d = np.load("predictions/file_102_file_107_fused_file_101_fused.npy", allow_pickle=True)
#keypoints_3d = np.load("temp_seq/file_101_file_102_alternated.npy", allow_pickle=True)
keypoints_3d = np.load('predictions/file_101_aligned.npy', allow_pickle=True)[230 + offset_1:550 + offset_1] / 13
keypoints_3d = keypoints_3d.astype(float) 
print(len(keypoints_3d))


h36m_pts = [(3,2), (2,1), (1, 0), (0, 4), (4, 5), (5, 6), \
    (13, 12), (12, 11), (11, 8), (8, 14), (14, 15), (15, 16), \
    (8, 9), (9, 10), (8, 7), (7, 0)]


upper = keypoints_3d[:, 2] - keypoints_3d[:, 3]  # Right
lower = keypoints_3d[:, 5] - keypoints_3d[:, 6]  # Left

norm_upper = np.linalg.norm(upper, axis=1)
norm_lower = np.linalg.norm(lower, axis=1)

plt.plot(norm_upper, label='Монокулярное предсказание', color="black", linestyle="--")
#plt.plot(norm_lower, label='Монокулярное предсказание', color="black", linestyle="--")
#plt.title('wdw')
plt.xlabel('Момент времени (кадр)')
plt.ylabel('Длина, в условных единицах')
#plt.title('wdw')
#plt.title('wdw')






plt.legend()
plt.show()