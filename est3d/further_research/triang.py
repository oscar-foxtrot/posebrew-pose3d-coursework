
"""
import cv2
import numpy as np
import json

# Load one frame from your video or image sequence to get size
cap = cv2.VideoCapture('./neurologist/file_469.mp4')
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to read frame for intrinsic estimation")

img_h, img_w = frame.shape[:2]

# Approximate focal length as the diagonal of the image in pixels (common heuristic)
focal_length = np.sqrt(img_w**2 + img_h**2)

# Principal point is usually at the image center
cx = img_w / 2
cy = img_h / 2

# Construct camera intrinsic matrix K
K = np.array([[focal_length, 0, cx],
              [0, focal_length, cy],
              [0, 0, 1]], dtype=np.float64)

file_number0 = 469
file_number1 = 474
with open(f'output_file{file_number0}/predictions/file_{file_number0}_toalpha_0.json', 'r') as f:
    pts1 = json.load(f)

with open(f'output_file{file_number1}/predictions/file_{file_number1}_toalpha_0.json', 'r') as f:
    pts2 = json.load(f)

# print("Camera intrinsic matrix K:\n", K)


frame_pts1 = np.array(pts1[0]['keypoints'], dtype=np.float32).reshape(26, 3)[:, 0:2]
frame_pts2 = np.array(pts2[0 + 54]['keypoints'], dtype=np.float32).reshape(26, 3)[:, 0:2]
# Now, assuming you have pts1 and pts2 (Nx2 arrays of matching keypoints), you can continue:
# E.g., pts1 and pts2 must be np.float32
E, mask = cv2.findEssentialMat(frame_pts1, frame_pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R, t, mask_pose = cv2.recoverPose(E, frame_pts1, frame_pts2, K)

print(R)
print(t)
print('#######')


frame_pts1 = np.array(pts1[100]['keypoints'], dtype=np.float32).reshape(26, 3)[:, 0:2]
frame_pts2 = np.array(pts2[100 + 54]['keypoints'], dtype=np.float32).reshape(26, 3)[:, 0:2]
# Now, assuming you have pts1 and pts2 (Nx2 arrays of matching keypoints), you can continue:
# E.g., pts1 and pts2 must be np.float32
E, mask = cv2.findEssentialMat(frame_pts1, frame_pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R, t, mask_pose = cv2.recoverPose(E, frame_pts1, frame_pts2, K)

print(R)
print(t)
print('#######')

frame_pts1 = np.array(pts1[200]['keypoints'], dtype=np.float32).reshape(26, 3)[:, 0:2]
frame_pts2 = np.array(pts2[200 + 54]['keypoints'], dtype=np.float32).reshape(26, 3)[:, 0:2]
# Now, assuming you have pts1 and pts2 (Nx2 arrays of matching keypoints), you can continue:
# E.g., pts1 and pts2 must be np.float32
E, mask = cv2.findEssentialMat(frame_pts1, frame_pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R, t, mask_pose = cv2.recoverPose(E, frame_pts1, frame_pts2, K)

print(R)
print(t)
print('#######')



# Assume frame_pts1 and frame_pts2 are Nx2 float32 arrays of matched keypoints
# K, R, t as before

# Projection matrices
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
P1 = K @ P1

P2 = np.hstack((R, t))
P2 = K @ P2

# Convert points to homogeneous (shape 2xN for triangulatePoints)
pts1_h = frame_pts1.T
pts2_h = frame_pts2.T

# Triangulate
points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)

# Convert to Euclidean coordinates
points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]

# Transpose to Nx3
points_3d = points_3d.T

print('3D points shape:', points_3d.shape)

print(points_3d)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed to enable 3D plotting in matplotlib

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# points_3d is Nx3: columns = X, Y, Z
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='o')

# Optional: set labels and axis limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Triangulated Points')

plt.show()
"""
"""
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Assume pts1, pts2, K are already loaded as in your code
# Also assume pts1 and pts2 length corresponds to the frame count (or adjust accordingly)


cap = cv2.VideoCapture('./neurologist/file_469.mp4')
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to read frame for intrinsic estimation")

img_h, img_w = frame.shape[:2]

# Approximate focal length as the diagonal of the image in pixels (common heuristic)
focal_length = np.sqrt(img_w**2 + img_h**2)

# Principal point is usually at the image center
cx = img_w / 2
cy = img_h / 2

# Construct camera intrinsic matrix K
K = np.array([[focal_length, 0, cx],
              [0, focal_length, cy],
              [0, 0, 1]], dtype=np.float64)

file_number0 = 469
file_number1 = 474
with open(f'output_file{file_number0}/predictions/file_{file_number0}_toalpha_0.json', 'r') as f:
    pts1 = json.load(f)

with open(f'output_file{file_number1}/predictions/file_{file_number1}_toalpha_0.json', 'r') as f:
    pts2 = json.load(f)


# Load your pts1, pts2, K as before...

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter([], [], [], c='b', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Triangulated Keypoints Animation')

all_points = []

from scipy.spatial.transform import Rotation as R_scipy

Rs = []
ts = []

for frame_idx in range(len(pts1)):
    frame_pts1 = np.array(pts1[frame_idx]['keypoints'], dtype=np.float32).reshape(26, 3)[:, 0:2]
    frame_pts2 = np.array(pts2[frame_idx + 54]['keypoints'], dtype=np.float32).reshape(26, 3)[:, 0:2]

    E, mask = cv2.findEssentialMat(frame_pts1, frame_pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R_mat, t_vec, mask_pose = cv2.recoverPose(E, frame_pts1, frame_pts2, K)

    Rs.append(R_mat)
    ts.append(t_vec.flatten())

# Average translation vector (simple mean)
t_avg = np.mean(ts, axis=0)

# Average rotation: convert each R to quaternion
quats = [R_scipy.from_matrix(r).as_quat() for r in Rs]  # xyzw

# Average quaternions by normalizing sum (simple approx)
quat_avg = np.mean(quats, axis=0)
quat_avg /= np.linalg.norm(quat_avg)

R_avg = R_scipy.from_quat(quat_avg).as_matrix()

# Now use R_avg and t_avg fixed for all frames in animation

for frame_idx in range(len(pts1)):
    frame_pts1 = np.array(pts1[frame_idx]['keypoints'], dtype=np.float32).reshape(26, 3)[:, 0:2]
    frame_pts2 = np.array(pts2[frame_idx + 54]['keypoints'], dtype=np.float32).reshape(26, 3)[:, 0:2]

    E, mask = cv2.findEssentialMat(frame_pts1, frame_pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(E, frame_pts1, frame_pts2, K)

    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K @ P1

    P2 = np.hstack((R, t))
    P2 = K @ P2

    pts1_h = frame_pts1.T
    pts2_h = frame_pts2.T

    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]
    points_3d = points_3d.T

    all_points.append(points_3d)

all_points = np.vstack(all_points)

x_min, y_min, z_min = np.min(all_points, axis=0) - 0.1
x_max, y_max, z_max = np.max(all_points, axis=0) + 0.1

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

def init():
    scatter._offsets3d = ([], [], [])
    return scatter,

def update(frame_idx):
    frame_pts1 = np.array(pts1[frame_idx]['keypoints'], dtype=np.float32).reshape(26, 3)[:, 0:2]
    frame_pts2 = np.array(pts2[frame_idx + 54]['keypoints'], dtype=np.float32).reshape(26, 3)[:, 0:2]

    # Fixed pose:
    R = R_avg
    t = t_avg.reshape(3,1)

    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K @ P1

    P2 = np.hstack((R, t))
    P2 = K @ P2

    pts1_h = frame_pts1.T
    pts2_h = frame_pts2.T

    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]
    points_3d = points_3d.T
    ax.set_xlim(-0.05, 0.05)
    ax.set_ylim(-0.05, 0.05)
    ax.set_zlim(-0.1, 1)
    scatter._offsets3d = (points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])

    return scatter,

total_frames = min(len(pts1), len(pts2) - 54)

anim = FuncAnimation(fig, update, frames=range(total_frames), init_func=init,
                     interval=8.33, blit=False)  # blit=False here!

plt.show()

import matplotlib.pyplot as plt
import numpy as np

def compute_angle(v1, v2):
    # Normalize vectors
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    # Dot product and clamp to [-1,1] to avoid numerical errors
    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

angles = []

for frame_idx in range(len(pts1)):
    keypoints = np.array(pts1[frame_idx]['keypoints'], dtype=np.float32).reshape(26, 3)[:, 0:2]  # (26, 2)

    v1 = keypoints[14] - keypoints[12]
    v2 = keypoints[16] - keypoints[14]

    angle = compute_angle(v1, v2)
    angles.append(angle)

angles = np.array(angles)

plt.figure(figsize=(10, 5))
plt.plot(angles, label='Angle between (12,14) and (14,16)')
plt.xlabel('Frame')
plt.ylabel('Angle (degrees)')
plt.title('Joint Angle over Frames')
plt.legend()
plt.grid(True)
plt.show()
"""
"""
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R_scipy

# --- Load video frame for intrinsic estimation ---
cap = cv2.VideoCapture('./neurologist/file_469.mp4')
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to read frame for intrinsic estimation")

img_h, img_w = frame.shape[:2]

# Approximate focal length as diagonal of image (pixels)
focal_length = np.sqrt(img_w**2 + img_h**2)
cx = img_w / 2
cy = img_h / 2

# Camera intrinsic matrix K
K = np.array([[focal_length, 0, cx],
              [0, focal_length, cy],
              [0, 0, 1]], dtype=np.float64)

# --- Load keypoints JSON ---
file_number0 = 469
file_number1 = 474
with open(f'output_file{file_number0}/predictions/file_{file_number0}_toalpha_0.json', 'r') as f:
    pts1 = json.load(f)

with open(f'output_file{file_number1}/predictions/file_{file_number1}_toalpha_0.json', 'r') as f:
    pts2 = json.load(f)

# --- Helper: smooth keypoints (moving average) ---
def smooth_keypoints(keypoints_list, window=5):
    smoothed = []
    for i in range(len(keypoints_list)):
        start = max(0, i - window//2)
        end = min(len(keypoints_list), i + window//2 + 1)
        avg_kp = np.mean(keypoints_list[start:end], axis=0)
        smoothed.append(avg_kp)
    return smoothed

# Extract raw 2D keypoints (just XY coords) from JSON for smoothing
pts1_2d_raw = [np.array(frame['keypoints'], dtype=np.float32).reshape(26, 3)[:, 0:2] for frame in pts1]
pts2_2d_raw = [np.array(frame['keypoints'], dtype=np.float32).reshape(26, 3)[:, 0:2] for frame in pts2]

pts1_2d = smooth_keypoints(pts1_2d_raw, window=5)
pts2_2d = smooth_keypoints(pts2_2d_raw, window=5)

# --- Estimate fixed camera pose (R, t) once over sample frames ---
sample_indices = np.linspace(0, len(pts1_2d)-1, num=10, dtype=int)

Rs = []
ts = []

for idx in sample_indices:
    frame_pts1 = pts1_2d[idx]
    frame_pts2 = pts2_2d[idx + 54]  # keep your +54 offset

    E, mask = cv2.findEssentialMat(frame_pts1, frame_pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        continue
    _, R_mat, t_vec, mask_pose = cv2.recoverPose(E, frame_pts1, frame_pts2, K)

    Rs.append(R_mat)
    ts.append(t_vec.flatten())

if len(Rs) == 0 or len(ts) == 0:
    raise RuntimeError("Pose estimation failed on sample frames.")

# Average translation vector
t_avg = np.mean(ts, axis=0)

# Average rotation quaternion
quats = [R_scipy.from_matrix(r).as_quat() for r in Rs]
quat_avg = np.mean(quats, axis=0)
quat_avg /= np.linalg.norm(quat_avg)
R_avg = R_scipy.from_quat(quat_avg).as_matrix()

# --- Triangulate all frames using fixed pose ---
all_points = []

for frame_idx in range(len(pts1_2d)):
    frame_pts1 = pts1_2d[frame_idx]
    frame_pts2 = pts2_2d[frame_idx + 54]

    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K @ P1

    P2 = np.hstack((R_avg, t_avg.reshape(3, 1)))
    P2 = K @ P2

    pts1_h = frame_pts1.T
    pts2_h = frame_pts2.T

    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]
    points_3d = points_3d.T

    all_points.append(points_3d)

all_points = np.vstack(all_points)

# Set plot limits based on 3D points spread
x_min, y_min, z_min = np.min(all_points, axis=0) - 0.1
x_max, y_max, z_max = np.max(all_points, axis=0) + 0.1

# --- Setup matplotlib 3D scatter animation ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter([], [], [], c='b', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Triangulated Keypoints Animation')

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

def init():
    scatter._offsets3d = ([], [], [])
    return scatter,

def update(frame_idx):
    frame_pts1 = pts1_2d[frame_idx]
    frame_pts2 = pts2_2d[frame_idx + 54]

    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K @ P1

    P2 = np.hstack((R_avg, t_avg.reshape(3, 1)))
    P2 = K @ P2

    pts1_h = frame_pts1.T
    pts2_h = frame_pts2.T

    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]
    points_3d = points_3d.T

    scatter._offsets3d = (points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
    return scatter,

total_frames = min(len(pts1_2d), len(pts2_2d) - 54)

anim = FuncAnimation(fig, update, frames=range(total_frames), init_func=init,
                     interval=8.33, blit=False)

plt.show()

# --- Compute joint angles from 2D keypoints for visualization ---

def compute_angle(v1, v2):
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

angles = []

for frame_idx in range(len(pts1_2d)):
    keypoints = pts1_2d[frame_idx]

    v1 = keypoints[14] - keypoints[12]
    v2 = keypoints[16] - keypoints[14]

    angle = compute_angle(v1, v2)
    angles.append(angle)

angles = np.array(angles)

plt.figure(figsize=(10, 5))
plt.plot(angles, label='Angle between (12,14) and (14,16)')
plt.xlabel('Frame')
plt.ylabel('Angle (degrees)')
plt.title('Joint Angle over Frames')
plt.legend()
plt.grid(True)
plt.show()
"""

import numpy as np
import cv2
import json
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R_scipy

# --- Load and utility functions ---

def load_keypoints(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def estimate_camera_intrinsics(frame):
    h, w = frame.shape[:2]
    f = np.sqrt(h**2 + w**2)
    cx, cy = w/2, h/2
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float64)
    return K

def recover_camera_poses(pts1_list, pts2_list, K, offset=0):
    Rs, ts = [], []
    n = len(pts1_list)
    for i in range(n):
        pts1 = np.array(pts1_list[i]['keypoints'], dtype=np.float32).reshape(26, 3)[:, :2]
        pts2 = np.array(pts2_list[i+offset]['keypoints'], dtype=np.float32).reshape(26, 3)[:, :2]

        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            continue
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
        Rs.append(R)
        ts.append(t.flatten())
    return Rs, ts

def average_pose(Rs, ts):
    t_avg = np.mean(ts, axis=0)
    quats = [R_scipy.from_matrix(R).as_quat() for R in Rs]
    quat_avg = np.mean(quats, axis=0)
    quat_avg /= np.linalg.norm(quat_avg)
    R_avg = R_scipy.from_quat(quat_avg).as_matrix()
    return R_avg, t_avg

def triangulate_points(pts1, pts2, K, R, t):
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K @ np.hstack((R, t.reshape(3,1)))
    pts1_h = pts1.T
    pts2_h = pts2.T
    points_4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    points_3d = (points_4d[:3] / points_4d[3]).T
    return points_3d

def reprojection_error(params, pts1, pts2, K, R, t):
    X = params.reshape(3,1)

    def project_point(X3D, P):
        X_hom = np.vstack((X3D, [[1]]))
        x_proj = P @ X_hom
        x_proj /= x_proj[2]
        return x_proj[:2].flatten()

    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K @ np.hstack((R, t.reshape(3,1)))

    proj1 = project_point(X, P1)
    proj2 = project_point(X, P2)

    residuals = np.hstack((proj1 - pts1, proj2 - pts2))
    return residuals

def refine_all_points(points_3d, pts1, pts2, K, R, t):
    refined_points = []
    for i in range(points_3d.shape[0]):
        x_init = points_3d[i]
        p1 = pts1[i]
        p2 = pts2[i]
        res = least_squares(reprojection_error, x_init, args=(p1, p2, K, R, t))
        refined_points.append(res.x)
    return np.array(refined_points)

# --- Load data ---

cap = cv2.VideoCapture('./neurologist/file_469.mp4')
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("Failed to read video frame")

K = estimate_camera_intrinsics(frame)

file_number0 = 469
file_number1 = 474
pts1 = load_keypoints(f'output_file{file_number0}/predictions/file_{file_number0}_toalpha_0.json')
pts2 = load_keypoints(f'output_file{file_number1}/predictions/file_{file_number1}_toalpha_0.json')

offset = 54

Rs, ts = recover_camera_poses(pts1, pts2, K, offset)
R_avg, t_avg = average_pose(Rs, ts)

# --- Triangulate and refine all frames ---

all_points_refined = []
for i in range(len(pts1)):
    pts1_frame = np.array(pts1[i]['keypoints'], dtype=np.float32).reshape(26, 3)[:, :2]
    pts2_frame = np.array(pts2[i+offset]['keypoints'], dtype=np.float32).reshape(26, 3)[:, :2]

    points_3d_init = triangulate_points(pts1_frame, pts2_frame, K, R_avg, t_avg)
    points_3d_refined = refine_all_points(points_3d_init, pts1_frame, pts2_frame, K, R_avg, t_avg)

    all_points_refined.append(points_3d_refined)

all_points_refined = np.array(all_points_refined)  # shape: (frames, 26, 3)

# --- 3D Animation Visualization ---

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter([], [], [], c='b', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Triangulated Keypoints Animation (Refined)')

x_min, y_min, z_min = np.min(all_points_refined.reshape(-1, 3), axis=0) - 0.1
x_max, y_max, z_max = np.max(all_points_refined.reshape(-1, 3), axis=0) + 0.1
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

def init():
    scatter._offsets3d = ([], [], [])
    return scatter,

def update(frame_idx):
    points_3d = all_points_refined[frame_idx]
    ax.set_xlim(-0.05, 0.05)
    ax.set_ylim(-0.05, 0.05)
    ax.set_zlim(-0.1, 1)
    scatter._offsets3d = (points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
    return scatter,

total_frames = all_points_refined.shape[0]

anim = FuncAnimation(fig, update, frames=range(total_frames), init_func=init,
                     interval=8.33, blit=False)

plt.show()

# --- Example: Compute and plot a joint angle over frames using refined 3D points ---

def compute_angle(v1, v2):
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    return np.degrees(angle_rad)

angles = []
for frame_idx in range(total_frames):
    pts_3d = all_points_refined[frame_idx]

    # Example: compute angle at keypoint 14 between vectors (12->14) and (14->16)
    v1 = pts_3d[13] - pts_3d[11]
    v2 = pts_3d[15] - pts_3d[13]

    angle = compute_angle(v1, v2)
    angles.append(angle)

plt.figure(figsize=(10, 5))
plt.plot(angles, label='Joint Angle (keypoints 11-13-15)')
plt.xlabel('Frame')
plt.ylabel('Angle (degrees)')
plt.title('Joint Angle over Frames (3D Refined)')
plt.legend()
plt.grid(True)
#plt.savefig('angles_triangulation_bad_2.png', dpi=200)
plt.show()

