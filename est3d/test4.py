import numpy as np
import json
file_number = 469

file_path_0 = f'D:/Programming/Programs_Py/est3d/output_test_{file_number}/predictions/file_{file_number}.json'

with open(f'D:/Programming/Programs_Py/est3d/output_test_{file_number}/predictions/file_{file_number}.json', 'r') as f:
    kpts = json.load(f)

keypoints = []
for frame in kpts:
    keypoints += [frame['instances'][0]['keypoints']]

keypoints = np.array(keypoints)

np.save(f'./file_{file_number}.npy', keypoints)