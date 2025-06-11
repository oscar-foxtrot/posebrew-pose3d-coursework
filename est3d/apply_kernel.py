import numpy as np
import os

def get_kpts(file_number):

    file_path_0 = f'D:/Programming/Programs_Py/est3d/MotionBERT/prediction_file_{file_number}_0/X3D.npy'
    kpts0 = np.load(file_path_0)
    
    file_path_1 = f'D:/Programming/Programs_Py/est3d/MotionBERT/prediction_file_{file_number}_1/X3D.npy'
    if not os.path.exists(file_path_1):
        kpts1 = None
    else:
        kpts1 = np.load(file_path_1)
    
    file_path_2 = f'D:/Programming/Programs_Py/est3d/MotionBERT/prediction_file_{file_number}_2/X3D.npy'
    if not os.path.exists(file_path_2):
        kpts2 = None
    else:
        kpts2 = np.load(file_path_2)

    def kernel(x, sigma=40, center=121):
        return np.exp(-1 / 2 * ((x % 243 - center) / sigma)**2)


    # It is expected for kpts0 to always exist
    kpts = []
    for i in range(kpts0.shape[0]):
        res = 0
        wghts = 0

        wghts += kernel(i, center=121)
        res += kpts0[i] * kernel(i, center=121)

        if i >= 81 and kpts1 is not None:
            wghts += kernel(i - 81, center=121)
            res += kpts1[i - 81] * kernel(i - 81, center=121)

        if i >= 162 and kpts2 is not None:
            wghts += kernel(i - 162, center=121)
            res += kpts2[i - 162] * kernel(i - 162, center=121)

        res /= wghts
        kpts += [res]
    
    return np.array(kpts)