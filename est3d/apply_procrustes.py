# Using two 3D keypoint files, get two 3D keypoint files BUT with keypoints aligned via procrustes

import numpy as np
import matplotlib.pyplot as plt
# import scipy
"""
kpts0 = np.load('file_469.npy', allow_pickle=True)[:]
kpts1 = np.load('file_474.npy', allow_pickle=True)[:]

if len(kpts0) > len(kpts1):
    kpts0, kpts1 = kpts1, kpts0

# Mirror
kpts0[:, :, 0] *= -1  # mirror x axis
kpts0[:, :, 2] *= -1  # mirror z axis

diff_len = len(kpts1) - len(kpts0)
norms = []

for i in range(diff_len + 1):
    diff = kpts1[i: len(kpts1) - diff_len + i] - kpts0
    norms += [np.linalg.norm(diff.reshape(-1))]


plt.plot(norms, label='Евклидова норма разности векторов,\nвырезано 100 последних кадров из видео 469')
plt.legend()
plt.show()
"""

"""
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

    '''
    norm_X = np.linalg.norm(X0)
    norm_Y = np.linalg.norm(Y0)

    X0 /= norm_X
    Y0 /= norm_Y
    '''

    U, _, Vt = np.linalg.svd(Y0.T @ X0)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    s = 1
    aligned_Y = s * Y0 @ R + X_mean

    return aligned_Y, R, s, X_mean - s * Y_mean @ R


import numpy as np
import matplotlib.pyplot as plt

kpts1 = np.load('file_469.npy', allow_pickle=True)[:]
kpts0 = np.load('file_474.npy', allow_pickle=True)[:]
kpts0 = np.array(kpts0, dtype=np.float64)
kpts1 = np.array(kpts1, dtype=np.float64)

if len(kpts0) > len(kpts1):
    kpts0, kpts1 = kpts1, kpts0

diff_len = len(kpts1) - len(kpts0)
norms = []

for i in range(diff_len + 1): 
    new_kpts0 = np.copy(kpts0)
    kpts1_new = kpts1[i: len(kpts1) - diff_len + i]
    
    for j in range(len(kpts0)):
        X = kpts1_new[j]  # Reference skeleton
        Y = new_kpts0[j]
        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)
        Y_aligned, R, scale, translation = similarity_procrustes(X, Y)
        new_kpts0[j] = Y_aligned

    diff = kpts1_new - new_kpts0
    diff = np.array(diff, dtype=np.float64)
    score = np.average(np.linalg.norm(diff, axis=2))
    norms += [score]


i = np.argmin(norms)
print(i)
plt.plot(norms, label='Среднее евклидова расстояния\nмежду точками (между кадрами)')
plt.legend()
#plt.savefig('With_alignment_With_scale_100.jpg')
plt.show()



kpts1_new = kpts1[i: len(kpts1) - diff_len + i]
for j in range(len(kpts0)):
    X = kpts1_new[j]  # Reference skeleton
    Y = kpts0[j]
    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)
    Y_aligned, R, scale, translation = similarity_procrustes(X, Y)
    kpts0[j] = Y_aligned

np.save('aligned_469.npy', kpts0)
np.save('kpts1_best_offset.npy', kpts1_new)
"""


# UNWEIGHTED ERROR

import numpy as np
import matplotlib.pyplot as plt

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

# --- Load keypoints
kpts1 = np.load('file_469_2_withoutnorm.npy', allow_pickle=True)[:]
kpts0 = np.load('file_474_2_withoutnorm.npy', allow_pickle=True)[:]
kpts0 = np.array(kpts0, dtype=np.float64)
kpts1 = np.array(kpts1, dtype=np.float64)

# --- Trim to equal length
if len(kpts0) > len(kpts1):
    kpts0, kpts1 = kpts1, kpts0

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

# --- Find best offset
i = np.argmin(norms)
print("Best alignment offset index:", i)

plt.plot(norms, label='Среднее евклидова расстояния\nмежду точками (между кадрами)')
plt.legend()
#plt.savefig('AVERAGE_WINDOWSIZE100_CUTBOTH.png', dpi=300)
plt.show()

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
np.save('469withoutnorm.npy', kpts0_aligned)
np.save('474withoutnorm.npy', kpts1_new)
