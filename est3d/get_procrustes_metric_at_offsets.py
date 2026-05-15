import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import json

# Ensure that the first file in a pair is always a shorter video (without the margins (see lines 32, 35))
# than the second file AND the person in that first video moves TOWARDS the camera


file_1 = "temp_seq/file_469.npy"
file_2 = "temp_seq/file_474.npy"
base_1 = os.path.splitext(os.path.basename(file_1))[0]
base_2 = os.path.splitext(os.path.basename(file_2))[0]


window_size = 100  # You can change this to any value ≥ 1



h36m_pts = [(3,2), (2,1), (1, 0), (0, 4), (4, 5), (5, 6), \
    (13, 12), (12, 11), (11, 8), (8, 14), (14, 15), (15, 16), \
    (8, 9), (9, 10), (8, 7), (7, 0)]

def halpe2h36m(x):
    T, V, C = x.shape
    y = np.zeros([T,17,C])
    y[:,0,:] = x[:,19,:]
    y[:,1,:] = x[:,12,:]
    y[:,2,:] = x[:,14,:]
    y[:,3,:] = x[:,16,:]
    y[:,4,:] = x[:,11,:]
    y[:,5,:] = x[:,13,:]
    y[:,6,:] = x[:,15,:]
    y[:,7,:] = (x[:,18,:] + x[:,19,:]) * 0.5
    y[:,8,:] = x[:,18,:]
    y[:,9,:] = x[:,0,:]
    y[:,10,:] = x[:,17,:]
    y[:,11,:] = x[:,5,:]
    y[:,12,:] = x[:,7,:]
    y[:,13,:] = x[:,9,:]
    y[:,14,:] = x[:,6,:]
    y[:,15,:] = x[:,8,:]
    y[:,16,:] = x[:,10,:]
    return y

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

    return aligned_Y, R, s, X_mean - s * R.T @ Y_mean

margins = 40

synced = False
# --- Load keypoints
if synced:
    kpts0 = np.load(file_1, allow_pickle=True)[:]
else:
    kpts0 = np.load(file_1, allow_pickle=True)[margins: -margins]
kpts1 = np.load(file_2, allow_pickle=True)[:]
kpts0 = np.array(kpts0, dtype=np.float64)
kpts1 = np.array(kpts1, dtype=np.float64)

diff_len = len(kpts1) - len(kpts0)

print(len(kpts0))
print(len(kpts1))

if synced:
    # skip offset search, files are synced
    i = 0
else:
    # run offset search and alignment
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

    # --- Find the best offset
    i = np.argmin(norms)
    print("Best alignment offset index:", i)

    plt.plot(norms)
    plt.show()


### ADDED
with open(f"./mm_output/output_{base_1}/predictions/{base_1}_toalpha_0.json", "r") as f:
    data = json.load(f)[margins: -margins]
keypoints_2d = np.array([np.array(data[frame_number]["keypoints"]).reshape(-1, 3) for frame_number in range(len(data))])
#confs = halpe2h36m(keypoints_2d[:, :, 2:3])
keypoints_2d = halpe2h36m(keypoints_2d[:, :, 0:2])

#np.save(f'predictions/{base_1}_2d_synced.npy', keypoints_2d)

with open(f"./mm_output/output_{base_2}/predictions/{base_2}_toalpha_0.json", "r") as f:
    data = json.load(f)[i: len(kpts1) - diff_len + i]
keypoints_2d = np.array([np.array(data[frame_number]["keypoints"]).reshape(-1, 3) for frame_number in range(len(data))])
#confs = halpe2h36m(keypoints_2d[:, :, 2:3])
keypoints_2d = halpe2h36m(keypoints_2d[:, :, 0:2])

#np.save(f'predictions/{base_2}_2d_synced.npy', keypoints_2d)



# --- Final alignment with best offset
kpts1_new = kpts1[i: len(kpts1) - diff_len + i]
aligned_acc = np.zeros_like(kpts0)
counts = np.zeros((len(kpts0), 1, 1))

for j in range(len(kpts0) - window_size + 1):
    X_win = kpts1_new[j:j + window_size].reshape(-1, 3)
    Y_win = kpts0[j:j + window_size].reshape(-1, 3)

    Y_win_aligned, R, scale, translation = similarity_procrustes(X_win, Y_win)

    #if j % 10 == 0:
    #    print(j)
    #    print(R)

    Y_win_aligned = Y_win_aligned.reshape(window_size, -1, 3)

    aligned_acc[j:j + window_size] += Y_win_aligned
    counts[j:j + window_size] += 1

counts[counts == 0] = 1
kpts0_aligned = aligned_acc / counts

#os.makedirs("predictions", exist_ok=True)

# --- Save results
#np.save(f'predictions/{base_1}_aligned.npy', kpts0_aligned)
#np.save(f'predictions/{base_2}_aligned.npy', kpts1_new)
d0 = len(kpts1_new) // 2

k = 0.00
weights = np.array([1 / (1 + np.exp(-k * (d - d0))) for d in range(len(kpts1_new))])
#weights = np.array([(1 / 2) for d in range(len(kpts1_new))])

#os.makedirs("predictions", exist_ok=True)

keypoints = np.array([kpts0_aligned[i] * weights[i] + kpts1_new[i] * (1 - weights[i]) for i in range(len(kpts1_new))])
#np.save(f'predictions/{base_1}_{base_2}_fused.npy', keypoints)


# Points may be animated next
#exit()
fig, ax = plt.subplots()

ax.plot(np.arange(len(norms)) - margins, norms, color='black')

ax.set_xlabel("Смещение начала видео 1 относительно начала видео 2")
ax.set_ylabel("Значение метрики")

ax.axvline(0, color='gray', linestyle='--', linewidth=1)

plt.show()