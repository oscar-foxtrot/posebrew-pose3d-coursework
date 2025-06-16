import json
import numpy as np
import matplotlib.pyplot as plt

# Load the JSON
with open('output_test_mbert_469/predictions/file_469.json', 'r') as f:
    data = json.load(f)

# Collect keypoints
keypoints_3d = []

for frame_data in data:
    instances = frame_data["instances"]
    if not instances:
        continue
    first_instance = instances[0]  # pick first person
    kps = first_instance["keypoints"]  # list of [x, y, z]
    keypoints_3d.append(kps)

# Convert to NumPy array
keypoints = np.array(keypoints_3d)  # shape: [num_frames, num_keypoints, 3]

keypoints = np.load('MotionBERT/file_469_COCO_0/X3D.npy', allow_pickle=True)
point_0 = 1
point_1 = 2

diff = keypoints[:, point_1, :] - keypoints[:, point_0, :]
diff0 = np.array(diff, dtype=np.float32)
dist0 = np.linalg.norm(diff0, axis=1)

point_0 = 2
point_1 = 3

diff = keypoints[:, point_1, :] - keypoints[:, point_0, :]
diff1 = np.array(diff, dtype=np.float32)
dist1 = np.linalg.norm(diff1, axis=1)

crosses_norms = np.linalg.norm(np.cross(diff0, diff1, axis=1), axis=1)
norm_multiples = dist0 * dist1

sin_angles = crosses_norms / norm_multiples
angles = np.arcsin(sin_angles)
angles_degrees = np.degrees(angles)

angles_degrees0 = angles_degrees


point_0 = 4
point_1 = 5

diff = keypoints[:, point_1, :] - keypoints[:, point_0, :]
diff0 = np.array(diff, dtype=np.float32)
dist0 = np.linalg.norm(diff0, axis=1)

point_0 = 5
point_1 = 6

diff = keypoints[:, point_1, :] - keypoints[:, point_0, :]
diff1 = np.array(diff, dtype=np.float32)
dist1 = np.linalg.norm(diff1, axis=1)

crosses_norms = np.linalg.norm(np.cross(diff0, diff1, axis=1), axis=1)
norm_multiples = dist0 * dist1

sin_angles = crosses_norms / norm_multiples
angles = np.arcsin(sin_angles)
angles_degrees = np.degrees(angles)

angles_degrees1 = angles_degrees

start = 0
end = 2000

offset = 18
start_offset = 0 if start + offset <= 0 else start + offset
end_offset = end + offset
new_end = start + len(dist1[start_offset: end_offset])

plt.plot(range(len(angles_degrees0))[start: end], angles_degrees0[start: end], label='Углы, правое колено')
#plt.plot(range(len(angles_degrees1))[start: end], angles_degrees1[start: end], label='Углы, левое колено')
plt.plot(range(len(angles_degrees1))[start_offset: end_offset], angles_degrees1[start: new_end], label=f'Углы, левое колено, сдвиг на {offset} кадров')

plt.legend()
#plt.savefig('knee_angles_offset_795.png', dpi=300)

plt.show()

plt.plot