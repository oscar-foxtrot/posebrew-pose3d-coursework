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
#keypoints_3d = np.load("temp_seq/file_101_file_102_alternated.npy", allow_pickle=True)
keypoints_3d = np.load('predictions/file_102_aligned.npy', allow_pickle=True)[230 + offset_1:550 + offset_1]
keypoints_3d = keypoints_3d.astype(float) 
print(len(keypoints_3d))


h36m_pts = [(3,2), (2,1), (1, 0), (0, 4), (4, 5), (5, 6), \
    (13, 12), (12, 11), (11, 8), (8, 14), (14, 15), (15, 16), \
    (8, 9), (9, 10), (8, 7), (7, 0)]


# right leg
upper = keypoints_3d[:, 1] - keypoints_3d[:, 2]
lower = keypoints_3d[:, 2] - keypoints_3d[:, 3]

npdot = np.sum(upper * lower, axis=1)
norm_upper = np.linalg.norm(upper, axis=1)
norm_lower = np.linalg.norm(lower, axis=1)

cosine_left = npdot / (norm_upper * norm_lower)

# left leg
upper = keypoints_3d[:, 4] - keypoints_3d[:, 5]
lower = keypoints_3d[:, 5] - keypoints_3d[:, 6]

npdot = np.sum(upper * lower, axis=1)
norm_upper = np.linalg.norm(upper, axis=1)
norm_lower = np.linalg.norm(lower, axis=1)

cosine_right = npdot / (norm_upper * norm_lower)

plt.plot(np.degrees(np.acos(cosine_left)), label='Угол в левом колене', color='black')
plt.plot(np.degrees(np.acos(cosine_right)), label='Угол в правом колене', color='black', linestyle='--')
#plt.title('wdw')
plt.xlabel('Момент времени (кадр)')
plt.ylabel('Угол (градусы)')




"""
#keypoints_3d = temp_seq/file_107.npy
#keypoints_3d = np.load("predictions/file_102_file_107_fused_file_101_fused.npy", allow_pickle=True)
#keypoints_3d = np.load("temp_seq/file_101_file_102_alternated.npy", allow_pickle=True)
keypoints_3d = np.load('predictions/file_101_aligned.npy', allow_pickle=True)[230 + offset_1:550 + offset_1]

keypoints_3d = keypoints_3d.astype(float) 
print(len(keypoints_3d))


# right leg
upper = keypoints_3d[:, 1] - keypoints_3d[:, 2]
lower = keypoints_3d[:, 2] - keypoints_3d[:, 3]

npdot = np.sum(upper * lower, axis=1)
norm_upper = np.linalg.norm(upper, axis=1)
norm_lower = np.linalg.norm(lower, axis=1)

cosine_left = npdot / (norm_upper * norm_lower)

# left leg
upper = keypoints_3d[:, 4] - keypoints_3d[:, 5]
lower = keypoints_3d[:, 5] - keypoints_3d[:, 6]

npdot = np.sum(upper * lower, axis=1)
norm_upper = np.linalg.norm(upper, axis=1)
norm_lower = np.linalg.norm(lower, axis=1)

cosine_right = npdot / (norm_upper * norm_lower)

plt.plot(np.degrees(np.acos(cosine_left)), label='Угол в левом колене', color="red", alpha=0.5)
plt.plot(np.degrees(np.acos(cosine_right)), label='Угол в правом колене', color="blue", alpha=0.5)
#plt.title('wdw')
"""






plt.legend()
plt.show()