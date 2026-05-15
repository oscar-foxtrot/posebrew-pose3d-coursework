import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------- SKELETON ----------------
h36m_pts = [
    (3,2), (2,1), (1,0), (0,4), (4,5), (5,6),
    (13,12), (12,11), (11,8), (8,14), (14,15), (15,16),
    (8,9), (9,10), (8,7), (7,0)
]

file_name_1 = "temp_seq/file_101_file_102_alternated.npy"
base_1 = "file_101"
base_2 = "file_102"

keypoints_3d = np.load(file_name_1, allow_pickle=True).astype(float)

# ---------------- CAMERA ----------------
R = np.array([
    [ 0.24088897, -0.01618557, -0.97041771],
    [-0.05622886,  0.99794879, -0.03060258],
    [ 0.96892251,  0.06193731,  0.23948476]
])

t = np.array([0.75218618, -0.01540476, 0.65877055]).reshape(3,1)

K1 = np.array([
    [1059.53, 0, 960],
    [0, 1059.53, 540],
    [0, 0, 1]
])

K2 = np.array([
    [1225.47, 0, 960],
    [0, 1225.47, 540],
    [0, 0, 1]
])

P1 = K1 @ np.hstack([np.eye(3), np.zeros((3,1))])
P2 = K2 @ np.hstack([R, t])

# ---------------- FRAME ----------------
FRAME_ID = 170

def project(P, X):
    Xh = np.hstack([X, np.ones((len(X), 1))])
    x = (P @ Xh.T).T
    return x[:, :2] / x[:, 2:3]

# ---------------- 3D FRAME ----------------
X = keypoints_3d[FRAME_ID]
x, y, z = X[:,0], X[:,1], X[:,2]

# ---------------- REPROJECTION ----------------
uv1 = project(P1, X)
uv2 = project(P2, X)

# ---------------- 2D DATA ----------------
pts_2d_1_all = np.load(f"predictions/{base_1}_2d_synced.npy", allow_pickle=True)
pts_2d_2_all = np.load(f"predictions/{base_2}_2d_synced.npy", allow_pickle=True)

pts_2d_1 = np.array(pts_2d_1_all, dtype=np.float64)[230:550]
pts_2d_2 = np.array(pts_2d_2_all, dtype=np.float64)[230:550]

frame = FRAME_ID

# ---------------- PLOT ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ---------- CAMERA 1 ----------
ax1.set_title("Вид с камеры 1")
ax1.set_xlim(0, 1920)
ax1.set_ylim(1080, 0)

ax1.scatter(
    pts_2d_1[frame][:,0],
    pts_2d_1[frame][:,1],
    color='red',
    label='Двумерное предсказание'
)

ax1.scatter(
    uv1[:,0],
    uv1[:,1],
    color='blue',
    label='Репроекция триангулированной позы'
)

for (i, j) in h36m_pts:
    ax1.plot(
        [pts_2d_1[frame][i,0], pts_2d_1[frame][j,0]],
        [pts_2d_1[frame][i,1], pts_2d_1[frame][j,1]],
        color='red'
    )
    ax1.plot(
        [uv1[i,0], uv1[j,0]],
        [uv1[i,1], uv1[j,1]],
        '--',
        color='blue'
    )

ax1.legend()

# ---------- CAMERA 2 ----------
ax2.set_title("Вид с камеры 2")
ax2.set_xlim(0, 1920)
ax2.set_ylim(1080, 0)

ax2.scatter(
    pts_2d_2[frame][:,0],
    pts_2d_2[frame][:,1],
    color='red',
    label='Двумерное предсказание'
)

ax2.scatter(
    uv2[:,0],
    uv2[:,1],
    color='blue',
    label='Репроекция триангулированной позы'
)

for (i, j) in h36m_pts:
    ax2.plot(
        [pts_2d_2[frame][i,0], pts_2d_2[frame][j,0]],
        [pts_2d_2[frame][i,1], pts_2d_2[frame][j,1]],
        color='red'
    )
    ax2.plot(
        [uv2[i,0], uv2[j,0]],
        [uv2[i,1], uv2[j,1]],
        '--',
        color='blue'
    )

ax1.set_aspect('equal', adjustable='box')
ax2.set_aspect('equal', adjustable='box')


ax2.legend()

plt.tight_layout()
plt.show()