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
# f = 1000
# varlambda = 0.75

print(len(keypoints_3d))
print(len(keypoints_2d))

varlambda = 0.9

f_estimate = estimate_global_f(
        keypoints_2d,
        keypoints_3d,
        skeleton_sums,
        varlambda,
        cx,
        cy,
        eps=1e-8
    )

print(f_estimate)

'''
estimates = []
for i in range(15):
    estimates.append(estimate_global_f(
        keypoints_2d[i * 30: i * 30 + 30],
        keypoints_3d[i * 30: i * 30 + 30],
        skeleton_sums[i * 30: i * 30 + 30],
        varlambda,
        cx,
        cy,
        eps=1e-8
    ))
plt.plot(estimates)
plt.show()
'''

f = f_estimate

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
scale = 10
ax.set_xlim([-2*scale, 0])
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
