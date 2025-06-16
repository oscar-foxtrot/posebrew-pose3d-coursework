import numpy as np

fused_pairs = [(469, 474), (727, 718)]


h36m_pts = [(3, 2), (2, 1), (1, 0), (0, 4), (4, 5), (5, 6), \
    (13, 12), (12, 11), (11, 8), (8, 14), (14, 15), (15, 16), \
    (8, 9), (9, 10), (8, 7), (7, 0)]
n = len(h36m_pts)

for pair_numbers in fused_pairs:

    first_file_path = f'D:/Programming/Programs_Py/est3d/preds/file_{pair_numbers[0]}_dynamic.npy'
    second_file_path = f'D:/Programming/Programs_Py/est3d/preds/file_{pair_numbers[1]}_dynamic.npy'
    fused_file_path = f'D:/Programming/Programs_Py/est3d/preds/{pair_numbers[0]}_{pair_numbers[1]}_fused_dynamic.npy'

    first_kpts = np.load(first_file_path, allow_pickle=True)
    second_kpts = np.load(second_file_path, allow_pickle=True)
    fused_kpts = np.load(fused_file_path, allow_pickle=True)

    M2 = first_kpts[1:]
    M1 = first_kpts[:-1]
    diffs = np.array(M2 - M1, dtype=np.float64)
    squared_norms = np.sum(diffs**2, axis=2)
    mean_squared = np.mean(squared_norms)
    rmsv_first = np.sqrt(mean_squared)

    T = len(first_kpts)
    d_t_ij = np.zeros((T, n))
    for t in range(T):
        frame = first_kpts[t]
        for k, (i, j) in enumerate(h36m_pts):
            p_i = frame[i]
            p_j = frame[j]
            dist = np.linalg.norm(p_i - p_j)
            d_t_ij[t, k] = dist
    var_per_edge = np.var(d_t_ij, axis=0)
    AV_first = np.sum(var_per_edge)


    M2 = second_kpts[1:]
    M1 = second_kpts[:-1]
    diffs = np.array(M2 - M1, dtype=np.float64)
    squared_norms = np.sum(diffs**2, axis=2)
    mean_squared = np.mean(squared_norms)
    rmsv_second = np.sqrt(mean_squared)

    T = len(second_kpts)
    d_t_ij = np.zeros((T, n))
    for t in range(T):
        frame = second_kpts[t]
        for k, (i, j) in enumerate(h36m_pts):
            p_i = frame[i]
            p_j = frame[j]
            dist = np.linalg.norm(p_i - p_j)
            d_t_ij[t, k] = dist
    var_per_edge = np.var(d_t_ij, axis=0)
    AV_second = np.sum(var_per_edge)


    M2 = fused_kpts[1:]
    M1 = fused_kpts[:-1]
    diffs = np.array(M2 - M1, dtype=np.float64)
    squared_norms = np.sum(diffs**2, axis=2)
    mean_squared = np.mean(squared_norms)
    rmsv_fused = np.sqrt(mean_squared)

    T = len(fused_kpts)
    d_t_ij = np.zeros((T, n))
    for t in range(T):
        frame = fused_kpts[t]
        for k, (i, j) in enumerate(h36m_pts):
            p_i = frame[i]
            p_j = frame[j]
            dist = np.linalg.norm(p_i - p_j)
            d_t_ij[t, k] = dist
    var_per_edge = np.var(d_t_ij, axis=0)
    AV_fused = np.sum(var_per_edge)


    print(f'{pair_numbers[0]} & {len(first_kpts)} & {rmsv_first:.5f} & {AV_first:.5f}\\\\')
    print(f'{pair_numbers[1]} & {len(second_kpts)} & {rmsv_second:.5f} & {AV_second:.5f}\\\\')
    print(f'{pair_numbers[0]}_{pair_numbers[1]}_fused & {len(fused_kpts)} & {rmsv_fused:.5f} & {AV_fused:.5f}\\\\')

