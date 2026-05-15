import numpy as np
import matplotlib.pyplot as plt
import json

# ---------------- SKELETON ----------------
h36m_pts = [
    (3,2), (2,1), (1,0), (0,4), (4,5), (5,6),
    (13,12), (12,11), (11,8), (8,14), (14,15), (15,16),
    (8,9), (9,10), (8,7), (7,0)
]

file_name_1 = "temp_seq/file_101_reproj_3d.npy"
keypoints_3d = np.load(file_name_1, allow_pickle=True).astype(float)



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

file_number = 101

with open(f"./mm_output/output_file_{file_number}/predictions/file_{file_number}_toalpha_0.json", "r") as f:
    data = json.load(f)

W, H = 1920, 1080
cx, cy = W / 2, H / 2

range_frames = range(0, 1726)
keypoints_2d = np.array([np.array(data[frame_number]["keypoints"]).reshape(-1, 3) for frame_number in range_frames])
keypoints_2d = halpe2h36m(keypoints_2d[:, :, 0:2])


print(len(keypoints_3d))
#exit()
# ---------------- FRAME ----------------
FRAME_ID = 310
X = keypoints_3d[FRAME_ID]
# ---------------- 2D DATA ----------------
pts_2d_1_all = keypoints_2d


pts_2d_1 = np.array(pts_2d_1_all, dtype=np.float64)

# ---------------- CAMERA ----------------
W, H = 1920, 1080
cx, cy = W / 2, H / 2

f = 614.78  # focal length

K = np.array([
    [f, 0, cx],
    [0, f, cy],
    [0, 0, 1]
])

P = K @ np.hstack([np.eye(3), np.zeros((3,1))])


# ---------------- PROJECTION ----------------
def project(P, X):
    Xh = np.hstack([X, np.ones((len(X), 1))])
    x = (P @ Xh.T).T
    return x[:, :2] / x[:, 2:3]


uv = project(P, X)


# ---------------- PLOT ----------------
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

#ax.set_title("Monocular Reprojection")
ax.set_xlim(0, 1920)
ax.set_ylim(1080, 0)

# ---------------- DETECTOR ----------------
ax.scatter(
    pts_2d_1[FRAME_ID][:,0],
    pts_2d_1[FRAME_ID][:,1],
    color='red',
    label='Двумерное предсказание'
)

# ---------------- REPROJECTION ----------------
ax.scatter(
    uv[:,0],
    uv[:,1],
    color='blue',
    label='Репроекция реконструированной трехмерной позы'
)

# ---------------- SKELETON ----------------
for (i, j) in h36m_pts:
    ax.plot(
        [pts_2d_1[FRAME_ID][i,0], pts_2d_1[FRAME_ID][j,0]],
        [pts_2d_1[FRAME_ID][i,1], pts_2d_1[FRAME_ID][j,1]],
        color='red'
    )

    ax.plot(
        [uv[i,0], uv[j,0]],
        [uv[i,1], uv[j,1]],
        '--',
        color='blue'
    )

ax.set_aspect('equal', adjustable='box')

ax.legend()
plt.tight_layout()
plt.show()