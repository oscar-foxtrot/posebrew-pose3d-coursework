import numpy as np

kpts = np.load('D:/Programming/Programs_Py/est3d/output_file826/predictions/file_826_bbox.npy', allow_pickle=True)
print(kpts.shape)
print(kpts[220])