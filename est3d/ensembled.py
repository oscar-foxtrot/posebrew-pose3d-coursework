import numpy as np
from apply_kernel import get_kpts

files = [356, 469, 474, 718, 727, 730, 734, 795, 826, 949, 1081]

for file_number in files:

    ######## NEW. GET KEYPOINTS
    keypoints = get_kpts(file_number)

    # Example 3D keypoints (replace with your actual data)
    keypoints_3d = keypoints

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

    from scipy.ndimage import gaussian_filter1d
    skeleton_sums = gaussian_filter1d(skeleton_sums, sigma=2)

    for i in range(len(keypoints_3d)):

        # Comment the lile below to remove normalization
        keypoints_3d[i] = (np.array(keypoints_3d[i]) / skeleton_sums[i]).tolist()

        for j in range(1, len(keypoints_3d[0])):
            keypoints_3d[i][j] = (np.array(keypoints_3d[i][j]) - np.array(keypoints_3d[i][0])).tolist()
        keypoints_3d[i][0] = [0, 0, 0]


    np.save(f'preds/file_{file_number}_single.npy', np.array(keypoints_3d, dtype=object))