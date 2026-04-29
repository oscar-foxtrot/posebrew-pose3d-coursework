import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from apply_kernel import get_kpts

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


import numpy as np

def estimate_global_f(
    keypoints_2d,
    keypoints_3d,
    skeleton_sums,
    varlambda,
    cx,
    cy,
    eps=1e-8
):
    A = 0.0
    B = 0.0

    T = len(keypoints_3d)

    for t in range(T):
        pose = np.asarray(keypoints_3d[t])
        u = keypoints_2d[t][:, 0]
        v = keypoints_2d[t][:, 1]

        S = 1 / skeleton_sums[t]

        # root 2D (you used this in your model)
        u0 = u[0]
        v0 = v[0]

        X = pose[:, 0]
        Y = pose[:, 1]
        Z = pose[:, 2]

        denom = Z + varlambda * S

        valid = np.abs(denom) > eps
        if np.sum(valid) < 2:
            continue

        a = X[valid] / denom[valid]
        b = Y[valid] / denom[valid]

        alpha = cx + (varlambda * S / denom[valid]) * (u0 - cx)
        beta  = cy + (varlambda * S / denom[valid]) * (v0 - cy)

        u_v = u[valid]
        v_v = v[valid]

        A += np.sum(a * a + b * b)
        B += np.sum((u_v - alpha) * a + (v_v - beta) * b)

    return B / (A + 1e-8)


file_number = 474

with open(f"./mm_output/output_file_{file_number}/predictions/file_{file_number}_toalpha_0.json", "r") as f:
    data = json.load(f)

W, H = 720, 1280
cx, cy = W / 2, H / 2

keypoints_3d = get_kpts(file_number)

h36m_pts = [(3,2), (2,1), (1, 0), (0, 4), (4, 5), (5, 6), \
    (13, 12), (12, 11), (11, 8), (8, 14), (14, 15), (15, 16), \
    (8, 9), (9, 10), (8, 7), (7, 0)]

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

skeleton_sums = gaussian_filter1d(skeleton_sums, sigma=2)

for i in range(len(keypoints_3d)):
    keypoints_3d[i] = (np.array(keypoints_3d[i]) / skeleton_sums[i]).tolist()
    for j in range(1, len(keypoints_3d[0])):
        keypoints_3d[i][j] = (np.array(keypoints_3d[i][j]) - np.array(keypoints_3d[i][0])).tolist()
    keypoints_3d[i][0] = [0, 0, 0]

#translations = []

#frame_number = 300

keypoints_2d = np.array([np.array(data[frame_number]["keypoints"]).reshape(-1, 3) for frame_number in range(len(keypoints_3d))])
keypoints_2d = halpe2h36m(keypoints_2d[:, :, 0:2])

# Now we know keypoints_3d (relative pose 3d), keypoints_2d (2d coordinates)

print(len(keypoints_3d))
print(len(keypoints_2d))

f = 975
varlambda = 0.75

for t in range(len(keypoints_3d)):

    Z0 = varlambda * 1.0 / skeleton_sums[t]

    u0, v0 = keypoints_2d[t][0]

    # root camera position (DIRECT BACKPROJECTION)
    X0 = (u0 - cx) / f * Z0
    Y0 = (v0 - cy) / f * Z0

    root = np.array([X0, Y0, Z0])

    # relative pose (already root-centered)
    X_rel = np.array(keypoints_3d[t])

    keypoints_3d[t] += root


def project_3d_to_2d(pose3d, f, cx, cy):
    X = pose3d[:, 0]
    Y = pose3d[:, 1]
    Z = pose3d[:, 2]

    valid = np.abs(Z) > 1e-8

    u = np.zeros_like(X)
    v = np.zeros_like(Y)

    u[valid] = cx + f * (X[valid] / Z[valid])
    v[valid] = cy + f * (Y[valid] / Z[valid])

    return np.stack([u, v], axis=-1)

keypoints_2d_pred = []

for t in range(len(keypoints_3d)):
    pose3d = np.array(keypoints_3d[t])
    uv = project_3d_to_2d(pose3d, f, cx, cy)
    keypoints_2d_pred.append(uv)

keypoints_2d_pred = np.array(keypoints_2d_pred)

fig, ax = plt.subplots(figsize=(6, 6))

ax.set_title("2D Original vs Reprojected")
ax.set_xlim(0, W)
ax.set_ylim(H, 0)  # image-style coordinates

orig_scatter = ax.scatter([], [], color="red", s=20, label="original")
proj_scatter = ax.scatter([], [], color="blue", s=20, label="reprojected")

ax.legend()

def update(frame):
    u_orig = keypoints_2d[frame][:, 0]
    v_orig = keypoints_2d[frame][:, 1]

    u_proj = keypoints_2d_pred[frame][:, 0]
    v_proj = keypoints_2d_pred[frame][:, 1]

    orig_scatter.set_offsets(np.stack([u_orig, v_orig], axis=1))
    proj_scatter.set_offsets(np.stack([u_proj, v_proj], axis=1))

    return orig_scatter, proj_scatter

ani = FuncAnimation(fig, update, frames=len(keypoints_3d), interval=33, blit=False)
plt.show()