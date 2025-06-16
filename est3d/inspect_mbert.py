import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
# keypoints = np.load('D:/Programming/Programs_Py/est3d/MotionBERT/file_949_pred0/X3D.npy')

# Print shape and a sample
'''
print("Shape:", keypoints.shape)
print("First frame keypoints:\n", keypoints[200])
print("Min values:", keypoints.min(axis=(0,1)))
print("Max values:", keypoints.max(axis=(0,1)))
'''


file_number = 795

######## NEW. GET KEYPOINTS
from apply_kernel import get_kpts
keypoints = get_kpts(file_number)

'''
### DELETE THIS
file_number = 469
file_path_0 = f'D:/Programming/Programs_Py/est3d/MotionBERT/TESTMESHTEST_{file_number}/regressed_keypoints3d.npy'
keypoints = np.load(file_path_0)
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation

# Example 3D keypoints (replace with your actual data)
keypoints_3d = keypoints

# Set up the figure and 3D axis
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Set up labels, limits, and titles
ax.set_title('3D Human Keypoints Animation')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

h36m_pts = [(3,2), (2,1), (1, 0), (0, 4), (4, 5), (5, 6), \
    (13, 12), (12, 11), (11, 8), (8, 14), (14, 15), (15, 16), \
    (8, 9), (9, 10), (8, 7), (7, 0)]
    

# Create scatter plot for keypoints and lines
scatter = ax.scatter([], [], [], color='blue')
lines = [ax.plot([], [], [], color='blue')[0] for _ in h36m_pts]

def get_ss_weighted(list_kpts_tuples, frame_kpts, weights):
    ss = 0
    for i in range(len(list_kpts_tuples)):
        kpt2_coords = frame_kpts[list_kpts_tuples[i][1]]
        kpt1_coords = frame_kpts[list_kpts_tuples[i][0]]
        ss += weights[i] * ((kpt2_coords[2] - kpt1_coords[2])**2 + (kpt2_coords[1] - kpt1_coords[1])**2 + (kpt2_coords[0] - kpt1_coords[0])**2)**0.5
    return ss

n = len(h36m_pts)
w0 = np.array([1 / (n)] * n)
skeleton_sums = []
for i in range(len(keypoints_3d)):
    skeleton_sums += [get_ss_weighted(h36m_pts, keypoints_3d[i], w0)]

"""
print(skeleton_sums)
plt.plot(range(len(skeleton_sums)), skeleton_sums)
plt.show()
"""

from scipy.ndimage import gaussian_filter1d
skeleton_sums = gaussian_filter1d(skeleton_sums, sigma=2)


for i in range(len(keypoints_3d)):

    # Comment the lile below to remove normalization
    
    keypoints_3d[i] = (np.array(keypoints_3d[i]) / skeleton_sums[i]).tolist()
    for j in range(1, len(keypoints_3d[0])):
        keypoints_3d[i][j] = (np.array(keypoints_3d[i][j]) - np.array(keypoints_3d[i][0])).tolist()
    keypoints_3d[i][0] = [0, 0, 0]
  

# print(keypoints_3d[300])

# Set axis limits based on the keypoints range
'''
ax.set_xlim([keypoints_3d[..., 0].min(), keypoints_3d[..., 0].max()])
ax.set_ylim([keypoints_3d[..., 1].min(), keypoints_3d[..., 1].max()])
ax.set_zlim([keypoints_3d[..., 2].min(), keypoints_3d[..., 2].max()])
'''
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])

'''
# Function to update the scatter plot for each frame
def update(frame):
    # Extract the (x, y, z) positions for all keypoints in the current frame
    x_vals = keypoints_3d[frame, :, 0]
    y_vals = keypoints_3d[frame, :, 1]
    z_vals = keypoints_3d[frame, :, 2]
    
    # Update the data for the scatter plot
    scatter._offsets3d = (z_vals, x_vals, -y_vals)
    
    return scatter,
'''
keypoints_3d = np.load('preds/file_469_dynamic.npy', allow_pickle=True)[:]
#keypoints_3d_0 = np.load('aligned_469.npy', allow_pickle=True)[40: -40]

#scatter2 = ax.scatter([], [], [], color='red')  # new scatter for second skeleton
#lines2 = [ax.plot([], [], [], color='red')[0] for _ in h36m_pts]

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
    '''
    x2 = keypoints_3d_0[frame, :, 0]
    y2 = keypoints_3d_0[frame, :, 1]
    z2 = keypoints_3d_0[frame, :, 2]
    scatter2._offsets3d = (-z2, x2, -y2)
    
    for line, (i, j) in zip(lines2, h36m_pts):
        line.set_data([-z2[i], -z2[j]], [x2[i], x2[j]])
        line.set_3d_properties([-y2[i], -y2[j]])
    '''
    return [scatter] + lines


#np.save('file_795_2_withoutnorm.npy', np.array(keypoints_3d, dtype=object))
# Animation

# Create the animation
ani = FuncAnimation(fig, update, frames=range(keypoints_3d.shape[0]), interval=8.33, blit=False)
#ani.save("file_795_NONORM.gif", writer="ffmpeg", fps=30)
# Show the plot
plt.show()

