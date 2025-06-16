
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load your keypoints as before
keypoints_3d = np.load('preds/469_474_fused_dynamic.npy', allow_pickle=True)[:]

h36m_pts = [(3,2), (2,1), (1, 0), (0, 4), (4, 5), (5, 6),
            (13, 12), (12, 11), (11, 8), (8, 14), (14, 15), (15, 16),
            (8, 9), (9, 10), (8, 7), (7, 0)]

frame = 300  # choose the frame index you want to visualize

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('Z')
ax.set_ylabel('X')
ax.set_zlabel('Y')

ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])

x_vals = keypoints_3d[frame, :, 0]
y_vals = keypoints_3d[frame, :, 1]
z_vals = keypoints_3d[frame, :, 2]


xticks = ax.get_xticks()
xlabels = [f"{-tick:.1f}" for tick in xticks]
ax.set_xticklabels(xlabels)

zticks = ax.get_zticks()
# Create labels as the negative of tick positions, formatted nicely
zlabels = [f"{-tick:.1f}" for tick in zticks]
# Set these labels on the x-axis
ax.set_zticklabels(zlabels)


# Scatter keypoints
ax.scatter(-z_vals, x_vals, -y_vals, color='green')

# Draw skeleton lines
for i, j in h36m_pts:
    ax.plot([-z_vals[i], -z_vals[j]], [x_vals[i], x_vals[j]], [-y_vals[i], -y_vals[j]], color='green')

plt.savefig('fused_combined_frame.png', dpi=200)
plt.show()


"""
import numpy as np

# Ensure that the first file in a pair is always a shorter video (without the margins (see lines 32, 35))
# than the second file AND the person in that first video moves TOWARDS the camera

pair_numbers = (469, 474)

window_size = 100  # You can change this to any value ≥ 1

def similarity_procrustes(X, Y):
    '''
    Procrustes analysis with scaling to align Y to X.
    Both X and Y are (N, 3) arrays.
    Returns: aligned_Y, rotation_matrix, scale, translation
    '''
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    X0 = X - X_mean
    Y0 = Y - Y_mean

    U, _, Vt = np.linalg.svd(Y0.T @ X0)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    s = 1  # No scaling, s = norm_X / norm_Y could be added here
    aligned_Y = s * Y0 @ R + X_mean

    return aligned_Y, R, s, X_mean - s * Y_mean @ R

margins = 50

# --- Load keypoints
kpts0 = np.load(f'preds/file_{pair_numbers[0]}_dynamic.npy', allow_pickle=True)[margins: -margins]
kpts1 = np.load(f'preds/file_{pair_numbers[1]}_dynamic.npy', allow_pickle=True)[:]
kpts0 = np.array(kpts0, dtype=np.float64)
kpts1 = np.array(kpts1, dtype=np.float64)

'''
# --- Trim to equal length
if len(kpts0) > len(kpts1):
    kpts0, kpts1 = kpts1, kpts0
'''

diff_len = len(kpts1) - len(kpts0)
norms = []

# --- Try different offsets
for i in range(diff_len + 1):
    kpts1_new = kpts1[i: len(kpts1) - diff_len + i]
    aligned_acc = np.zeros_like(kpts0)
    counts = np.zeros((len(kpts0), 1, 1))

    for j in range(len(kpts0) - window_size + 1):
        X_win = kpts1_new[j:j + window_size].reshape(-1, 3)
        Y_win = kpts0[j:j + window_size].reshape(-1, 3)

        Y_win_aligned, R, scale, translation = similarity_procrustes(X_win, Y_win)
        Y_win_aligned = Y_win_aligned.reshape(window_size, -1, 3)

        aligned_acc[j:j + window_size] += Y_win_aligned
        counts[j:j + window_size] += 1

    counts[counts == 0] = 1
    new_kpts0 = aligned_acc / counts

    diff = kpts1_new - new_kpts0
    diff = np.array(diff, dtype=np.float64)
    score = np.average(np.linalg.norm(diff, axis=2))
    norms.append(score)

# USE ONLY IF MARGINS = 50!!!
# --- Find best offset
i = np.argmin(norms)
if pair_numbers[0] == 469:
    i = 104
elif pair_numbers[0] == 727:
    i = 12
print("Best alignment offset index:", i)


# --- Final alignment with best offset
kpts1_new = kpts1[i: len(kpts1) - diff_len + i]
aligned_acc = np.zeros_like(kpts0)
counts = np.zeros((len(kpts0), 1, 1))

for j in range(len(kpts0) - window_size + 1):
    X_win = kpts1_new[j:j + window_size].reshape(-1, 3)
    Y_win = kpts0[j:j + window_size].reshape(-1, 3)

    Y_win_aligned, R, scale, translation = similarity_procrustes(X_win, Y_win)
    Y_win_aligned = Y_win_aligned.reshape(window_size, -1, 3)

    aligned_acc[j:j + window_size] += Y_win_aligned
    counts[j:j + window_size] += 1

counts[counts == 0] = 1
kpts0_aligned = aligned_acc / counts

# --- Save results
#np.save(f'{pair_numbers[0]}_aligned_dynamic.npy', kpts0_aligned)
#np.save(f'{pair_numbers[1]}_aligned_dynamic.npy', kpts1_new)
d0 = len(kpts1_new) // 2

k = 0.00
weights = np.array([1 / (1 + np.exp(-k * (d - d0))) for d in range(len(kpts1_new))])
#weights = np.array([(1 / 2) for d in range(len(kpts1_new))])

#keypoints = np.array([kpts0_aligned[i] * weights[i] + kpts1_new[i] * (1 - weights[i]) for i in range(len(kpts1_new))])

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load your keypoints as before
keypoints_3d = kpts0_aligned

h36m_pts = [(3,2), (2,1), (1, 0), (0, 4), (4, 5), (5, 6),
            (13, 12), (12, 11), (11, 8), (8, 14), (14, 15), (15, 16),
            (8, 9), (9, 10), (8, 7), (7, 0)]

frame = 300  # choose the frame index you want to visualize

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('Z')
ax.set_ylabel('X')
ax.set_zlabel('Y')

ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])

xticks = ax.get_xticks()
xlabels = [f"{-tick:.1f}" for tick in xticks]
ax.set_xticklabels(xlabels)

zticks = ax.get_zticks()
# Create labels as the negative of tick positions, formatted nicely
zlabels = [f"{-tick:.1f}" for tick in zticks]
# Set these labels on the x-axis
ax.set_zticklabels(zlabels)


x_vals = keypoints_3d[frame, :, 0]
y_vals = keypoints_3d[frame, :, 1]
z_vals = keypoints_3d[frame, :, 2]

# Scatter keypoints
ax.scatter(-z_vals, x_vals, -y_vals, color='blue')

# Draw skeleton lines
for i, j in h36m_pts:
    ax.plot([-z_vals[i], -z_vals[j]], [x_vals[i], x_vals[j]], [-y_vals[i], -y_vals[j]], color='blue')


keypoints_3d = kpts1_new
x_vals = keypoints_3d[frame, :, 0]
y_vals = keypoints_3d[frame, :, 1]
z_vals = keypoints_3d[frame, :, 2]

# Scatter keypoints
ax.scatter(-z_vals, x_vals, -y_vals, color='red')

# Draw skeleton lines
for i, j in h36m_pts:
    ax.plot([-z_vals[i], -z_vals[j]], [x_vals[i], x_vals[j]], [-y_vals[i], -y_vals[j]], color='red')

plt.savefig('fused_frame.png', dpi=200)
plt.show()
"""