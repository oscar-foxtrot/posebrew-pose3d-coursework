import numpy as np

# Ensure that the first file in a pair is always a shorter video (without the margins (see lines 32, 35))
# than the second file AND the person in that first video moves TOWARDS the camera

pairs_file_numbers = [(469, 474), (727, 718)]

window_size = 100  # You can change this to any value ≥ 1

def similarity_procrustes(X, Y):
    '''
    Procrustes analysis with scaling to align Y to X.
    Both X and Y are (N, 3) arrays.
    Returns: aligned_Y, rotation_matrix, scale, translation
    '''
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    X0 = X - X_mean
    Y0 = Y - Y_mean

    U, _, Vt = np.linalg.svd(Y0.T @ X0)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    s = 1  # No scaling, s = norm_X / norm_Y could be added here
    aligned_Y = s * Y0 @ R + X_mean

    return aligned_Y, R, s, X_mean - s * Y_mean @ R

margins = 50
for pair_numbers in pairs_file_numbers:
    # --- Load keypoints
    kpts0 = np.load(f'preds/file_{pair_numbers[0]}_dynamic.npy', allow_pickle=True)[margins: -margins]
    kpts1 = np.load(f'preds/file_{pair_numbers[1]}_dynamic.npy', allow_pickle=True)[:]
    kpts0 = np.array(kpts0, dtype=np.float64)
    kpts1 = np.array(kpts1, dtype=np.float64)

    """
    # --- Trim to equal length
    if len(kpts0) > len(kpts1):
        kpts0, kpts1 = kpts1, kpts0
    """

    diff_len = len(kpts1) - len(kpts0)
    norms = []

    # --- Try different offsets
    for i in range(diff_len + 1):
        kpts1_new = kpts1[i: len(kpts1) - diff_len + i]
        aligned_acc = np.zeros_like(kpts0)
        counts = np.zeros((len(kpts0), 1, 1))

        for j in range(len(kpts0) - window_size + 1):
            X_win = kpts1_new[j:j + window_size].reshape(-1, 3)
            Y_win = kpts0[j:j + window_size].reshape(-1, 3)

            Y_win_aligned, R, scale, translation = similarity_procrustes(X_win, Y_win)
            Y_win_aligned = Y_win_aligned.reshape(window_size, -1, 3)

            aligned_acc[j:j + window_size] += Y_win_aligned
            counts[j:j + window_size] += 1

        counts[counts == 0] = 1
        new_kpts0 = aligned_acc / counts

        diff = kpts1_new - new_kpts0
        diff = np.array(diff, dtype=np.float64)
        score = np.average(np.linalg.norm(diff, axis=2))
        norms.append(score)

    # --- Find best offset
    i = np.argmin(norms)
    if pair_numbers[0] == 469:
        i = 104
    elif pair_numbers[0] == 727:
        i = 12
    print("Best alignment offset index:", i)

    # --- Final alignment with best offset
    kpts1_new = kpts1[i: len(kpts1) - diff_len + i]
    aligned_acc = np.zeros_like(kpts0)
    counts = np.zeros((len(kpts0), 1, 1))

    for j in range(len(kpts0) - window_size + 1):
        X_win = kpts1_new[j:j + window_size].reshape(-1, 3)
        Y_win = kpts0[j:j + window_size].reshape(-1, 3)

        Y_win_aligned, R, scale, translation = similarity_procrustes(X_win, Y_win)
        Y_win_aligned = Y_win_aligned.reshape(window_size, -1, 3)

        aligned_acc[j:j + window_size] += Y_win_aligned
        counts[j:j + window_size] += 1

    counts[counts == 0] = 1
    kpts0_aligned = aligned_acc / counts

    # --- Save results
    #np.save(f'{pair_numbers[0]}_aligned_dynamic.npy', kpts0_aligned)
    #np.save(f'{pair_numbers[1]}_aligned_dynamic.npy', kpts1_new)
    d0 = len(kpts1_new) // 2

    k = 0.00
    weights = np.array([1 / (1 + np.exp(-k * (d - d0))) for d in range(len(kpts1_new))])
    #weights = np.array([(1 / 2) for d in range(len(kpts1_new))])

    keypoints = np.array([kpts0_aligned[i] * weights[i] + kpts1_new[i] * (1 - weights[i]) for i in range(len(kpts1_new))])
    np.save(f'preds/{pair_numbers[0]}_{pair_numbers[1]}_fused_dynamic.npy', keypoints)