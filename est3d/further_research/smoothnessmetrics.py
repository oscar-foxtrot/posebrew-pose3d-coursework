import numpy as np

files = [356, 469, 474, 718, 727, 730, 734, 795, 826, 949, 1081]
h36m_pts = [(3, 2), (2, 1), (1, 0), (0, 4), (4, 5), (5, 6), \
    (13, 12), (12, 11), (11, 8), (8, 14), (14, 15), (15, 16), \
    (8, 9), (9, 10), (8, 7), (7, 0)]
n = len(h36m_pts)

l = 0

for file_number in files:
    l += 1

    file_path = f'D:/Programming/Programs_Py/est3d/preds/file_{file_number}_single.npy'
    kpts = np.load(file_path, allow_pickle=True)

    M2 = kpts[1:]
    M1 = kpts[:-1]
    diffs = np.array(M2 - M1, dtype=np.float64)
    squared_norms = np.sum(diffs**2, axis=2)
    mean_squared = np.mean(squared_norms)
    rmsv_single = np.sqrt(mean_squared)

    T = len(kpts)
    d_t_ij = np.zeros((T, n))
    for t in range(T):
        frame = kpts[t]
        for k, (i, j) in enumerate(h36m_pts):
            p_i = frame[i]
            p_j = frame[j]
            dist = np.linalg.norm(p_i - p_j)
            d_t_ij[t, k] = dist
    var_per_edge = np.var(d_t_ij, axis=0)
    AV_single = np.sum(var_per_edge)


    file_path = f'D:/Programming/Programs_Py/est3d/preds/file_{file_number}_uniform.npy'
    kpts = np.load(file_path, allow_pickle=True)

    M2 = kpts[1:]
    M1 = kpts[:-1]
    diffs = np.array(M2 - M1, dtype=np.float64)
    squared_norms = np.sum(diffs**2, axis=2)
    mean_squared = np.mean(squared_norms)
    rmsv_uniform = np.sqrt(mean_squared)

    T = len(kpts)
    d_t_ij = np.zeros((T, n))
    for t in range(T):
        frame = kpts[t]
        for k, (i, j) in enumerate(h36m_pts):
            p_i = frame[i]
            p_j = frame[j]
            dist = np.linalg.norm(p_i - p_j)
            d_t_ij[t, k] = dist
    var_per_edge = np.var(d_t_ij, axis=0)
    AV_uniform = np.sum(var_per_edge)


    file_path = f'D:/Programming/Programs_Py/est3d/preds/file_{file_number}_dynamic.npy'
    kpts = np.load(file_path, allow_pickle=True)

    M2 = kpts[1:]
    M1 = kpts[:-1]
    diffs = np.array(M2 - M1, dtype=np.float64)
    squared_norms = np.sum(diffs**2, axis=2)
    mean_squared = np.mean(squared_norms)
    rmsv_dynamic = np.sqrt(mean_squared)

    T = len(kpts)
    d_t_ij = np.zeros((T, n))
    for t in range(T):
        frame = kpts[t]
        for k, (i, j) in enumerate(h36m_pts):
            p_i = frame[i]
            p_j = frame[j]
            dist = np.linalg.norm(p_i - p_j)
            d_t_ij[t, k] = dist
    var_per_edge = np.var(d_t_ij, axis=0)
    AV_dynamic = np.sum(var_per_edge)

    print(f'{l} & {rmsv_single:.5f} & {rmsv_uniform:.5f} & {rmsv_dynamic:.5f} & '
        f'{AV_single:.5f} & {AV_uniform:.5f} & {AV_dynamic:.5f}\\\\')

