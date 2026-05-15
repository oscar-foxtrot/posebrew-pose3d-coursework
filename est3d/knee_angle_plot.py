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
keypoints_3d = np.load("temp_seq/file_101.npy", allow_pickle=True)

keypoints_3d = keypoints_3d.astype(float) 
print(len(keypoints_3d))

start = 302
range_frames = range(start, start + 520)
keypoints_3d = keypoints_3d[range_frames]

h36m_pts = [(3,2), (2,1), (1, 0), (0, 4), (4, 5), (5, 6), \
    (13, 12), (12, 11), (11, 8), (8, 14), (14, 15), (15, 16), \
    (8, 9), (9, 10), (8, 7), (7, 0)]



"""
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_title('3D Human Keypoints Animation')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


# Create scatter plot for keypoints and lines
scatter = ax.scatter([], [], [], color='blue')
lines = [ax.plot([], [], [], color='blue')[0] for _ in h36m_pts]
  
# Set axis limits based on the keypoints range
scale = 2
ax.set_xlim([-scale, scale])
ax.set_ylim([-scale, scale])
ax.set_zlim([-scale, scale])

def update(frame):
    x_vals = keypoints_3d[frame, :, 0]
    y_vals = keypoints_3d[frame, :, 1]
    z_vals = keypoints_3d[frame, :, 2]
    
    # Update scatter plot
    scatter._offsets3d = (-z_vals, x_vals, -y_vals)

    # Update line segments
    for line, (i, j) in zip(lines, h36m_pts):
        line.set_data([-z_vals[i], -z_vals[j]], [x_vals[i], x_vals[j]])
        line.set_3d_properties([-y_vals[i], -y_vals[j]])

    return [scatter] + lines


ani = FuncAnimation(fig, update, frames=range(keypoints_3d.shape[0]), interval=8.33, blit=False)
#ani.save("file_795_NONORM.gif", writer="ffmpeg", fps=30)
# Show the plot
plt.show()
"""
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

plt.plot(np.degrees(np.acos(cosine_left)), label='Угол в левом колене')
plt.plot(np.degrees(np.acos(cosine_right)), label='Угол в правом колене')
#plt.title('wdw')

plt.legend()
plt.show()