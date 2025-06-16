'''
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load JSON data
with open('output_test_mbert_469/predictions/file_469.json', 'r') as f:
    data = json.load(f)

# Extract 3D keypoints [x, y, z] per instance per frame
frames = []
for frame_data in data:
    frame = []
    for instance in frame_data["instances"]:
        kps = instance["keypoints"]
        frame.append(kps)
    frames.append(frame)

# Set up the 2D plot (side view: x vs z)
fig, ax = plt.subplots()
scatters = []

def init():
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Z (Depth)")
    return scatters

# Update function
def update(frame_idx):
    ax.cla()
    init()
    frame = frames[frame_idx]
    for instance in frame:
        x = [kp[1] for kp in instance]
        z = [kp[2] for kp in instance]  # Using depth for vertical axis
        sc = ax.scatter(x, z, s=30)
        scatters.append(sc)
    ax.set_title(f"Frame {frame_idx}")
    return scatters

# Create animation
ani = FuncAnimation(fig, update, frames=len(frames), init_func=init, blit=False, interval=8.33)

plt.show()
'''

import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load JSON data
with open('output_test_mbert_469/predictions/file_469.json', 'r') as f:
    data = json.load(f)

# Extract 3D keypoints [x, y, z] per instance per frame
frames = []
for frame_data in data:
    frame = []
    for instance in frame_data["instances"]:
        kps = instance["keypoints"]
        frame.append(kps)
    frames.append(frame)

# Human3.6M skeleton connections
h36m_pts = [(3,2), (2,1), (1,0), (0,4), (4,5), (5,6),
            (13,12), (12,11), (11,8), (8,14), (14,15), (15,16),
            (8,9), (9,10), (8,7), (7,0)]

# Set up the 2D plot (side view: Y vs Z, i.e., side view)
fig, ax = plt.subplots()
scatters = []

def init():
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("Y (Side View)")
    ax.set_ylabel("Z (Depth)")
    return scatters

# Update function
def update(frame_idx):
    ax.cla()
    init()
    frame = frames[frame_idx]
    for instance in frame:
        y = [kp[1] for kp in instance]  # Y axis becomes horizontal
        z = [kp[2] for kp in instance]  # Z axis (depth) becomes vertical
        sc = ax.scatter(y, z, s=30, c='blue')
        scatters.append(sc)

        # Draw skeleton lines
        for i, j in h36m_pts:
            if i < len(instance) and j < len(instance):
                xi, zi = instance[i][1], instance[i][2]
                xj, zj = instance[j][1], instance[j][2]
                ax.plot([xi, xj], [zi, zj], c='black', linewidth=1)
    ax.set_title(f"Frame {frame_idx}")
    return scatters

# Create animation
ani = FuncAnimation(fig, update, frames=len(frames), init_func=init, blit=False, interval=33.33)
ani.save('anim_MMPOSE_MB.gif', writer='Pillow')

plt.show()