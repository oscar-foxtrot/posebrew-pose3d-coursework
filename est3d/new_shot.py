import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation as Rot
import os
from scipy.ndimage import gaussian_filter1d

from apply_kernel import get_kpts

# ============================================================
# Utilities
# ============================================================

def to_homogeneous(x: np.ndarray) -> np.ndarray:
    """(N,2) -> (N,3)"""
    return np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)


def normalize_points_2d(x: np.ndarray):
    """
    Hartley normalization.
    x: (N,2)
    Returns:
        x_norm: (N,2)
        T: (3,3)
    """
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=0)
    xc = x - mu
    mean_dist = np.mean(np.sqrt(np.sum(xc ** 2, axis=1)))
    s = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0

    T = np.array([
        [s, 0.0, -s * mu[0]],
        [0.0, s, -s * mu[1]],
        [0.0, 0.0, 1.0],
    ])

    xh = to_homogeneous(x)
    xnh = (T @ xh.T).T
    x_norm = xnh[:, :2] / xnh[:, 2:3]
    return x_norm, T


def build_camera_matrix(f: float, cx: float, cy: float, R: np.ndarray = None, t: np.ndarray = None):
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)
    K = np.array([
        [f, 0.0, cx],
        [0.0, f, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    P = K @ np.hstack([R, t.reshape(3, 1)])
    return K, P


# ============================================================
# Fundamental matrix estimation (normalized 8-point + RANSAC)
# ============================================================

def estimate_F_8point(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    x1, x2: (N,2), N>=8
    Returns F with rank-2 enforced.
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    assert x1.shape == x2.shape and x1.shape[0] >= 8

    x1n, T1 = normalize_points_2d(x1)
    x2n, T2 = normalize_points_2d(x2)

    X = x1n[:, 0]
    Y = x1n[:, 1]
    Xp = x2n[:, 0]
    Yp = x2n[:, 1]

    A = np.column_stack([
        Xp * X,
        Xp * Y,
        Xp,
        Yp * X,
        Yp * Y,
        Yp,
        X,
        Y,
        np.ones_like(X),
    ])

    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0.0
    F = U @ np.diag(S) @ Vt

    # denormalize
    F = T2.T @ F @ T1
    if abs(F[2, 2]) > 1e-12:
        F = F / F[2, 2]
    else:
        F = F / np.linalg.norm(F)
    return F


def sampson_distance(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    x1, x2: (N,2)
    Returns per-point Sampson distance.
    """
    x1h = to_homogeneous(np.asarray(x1, dtype=np.float64))
    x2h = to_homogeneous(np.asarray(x2, dtype=np.float64))

    Fx1 = (F @ x1h.T).T
    Ftx2 = (F.T @ x2h.T).T
    x2tFx1 = np.sum(x2h * Fx1, axis=1)
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return (x2tFx1 ** 2) / np.maximum(denom, 1e-12)


def ransac_F(
    x1: np.ndarray,
    x2: np.ndarray,
    n_iter: int = 3000,
    thresh: float = 1.5,
    rng: np.random.Generator = None,
):
    """
    Minimal RANSAC for F.
    x1, x2: (N,2)
    Returns:
        F_best, inlier_mask
    """
    if rng is None:
        rng = np.random.default_rng(0)

    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    N = x1.shape[0]
    assert N >= 8

    best_inliers = None
    best_F = None
    best_score = -1

    idx_all = np.arange(N)

    for _ in range(n_iter):
        sample = rng.choice(idx_all, size=8, replace=False)
        try:
            F = estimate_F_8point(x1[sample], x2[sample])
        except np.linalg.LinAlgError:
            continue

        d = sampson_distance(F, x1, x2)
        inliers = d < thresh ** 2
        score = int(np.sum(inliers))
        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_F = F

    if best_F is None:
        raise RuntimeError("RANSAC failed to find a fundamental matrix")

    # Refit on all inliers
    F_refit = estimate_F_8point(x1[best_inliers], x2[best_inliers])
    return F_refit, best_inliers


# ============================================================
# Essential matrix / camera decomposition
# ============================================================

def K_from_f(f: float, cx: float, cy: float) -> np.ndarray:
    return np.array([
        [f, 0.0, cx],
        [0.0, f, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def essential_from_F(F: np.ndarray, f1: float, f2: float, cx: float, cy: float) -> np.ndarray:
    K1 = K_from_f(f1, cx, cy)
    K2 = K_from_f(f2, cx, cy)
    E = K2.T @ F @ K1
    # scale-normalize for numerical stability
    n = np.linalg.norm(E)
    if n > 1e-12:
        E = E / n
    return E


def decompose_E(E: np.ndarray):
    """
    Return candidate (R, t) pairs from essential matrix.
    t is unit norm up to sign.
    """
    U, _, Vt = np.linalg.svd(E)

    # Ensure proper rotations
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    W = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    # fix reflections if needed
    if np.linalg.det(R1) < 0:
        R1 = -R1
        t = -t
    if np.linalg.det(R2) < 0:
        R2 = -R2

    candidates = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t),
    ]
    return candidates


# ============================================================
# Triangulation and cheirality
# ============================================================

def triangulate_points_linear(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Linear DLT triangulation.
    x1, x2: (N,2)
    Returns X: (N,3)
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    N = x1.shape[0]
    X = np.zeros((N, 3), dtype=np.float64)

    for i in range(N):
        u1, v1 = x1[i]
        u2, v2 = x2[i]
        A = np.array([
            u1 * P1[2] - P1[0],
            v1 * P1[2] - P1[1],
            u2 * P2[2] - P2[0],
            v2 * P2[2] - P2[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        Xh = Vt[-1]
        X[i] = Xh[:3] / Xh[3]
    return X


def cheirality_count(P1: np.ndarray, P2: np.ndarray, X: np.ndarray) -> int:
    Xh = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    z1 = (P1[2] @ Xh.T)
    z2 = (P2[2] @ Xh.T)
    return int(np.sum((z1 > 0) & (z2 > 0)))


# ============================================================
# MotionBERT prior: similarity alignment (Procrustes)
# ============================================================

def procrustes_similarity(X: np.ndarray, Y: np.ndarray):
    """
    Find s, R, t minimizing || X - (s R Y + t) ||_F^2
    X, Y: (J,3)
    Returns aligned_Y, (s, R, t), error
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    X0 = X - muX
    Y0 = Y - muY

    H = Y0.T @ X0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    varY = np.sum(Y0 ** 2)
    s = np.sum(S) / max(varY, 1e-12)
    t = muX - s * (R @ muY)

    Y_aligned = (s * (R @ Y.T)).T + t
    err = np.mean(np.sum((X - Y_aligned) ** 2, axis=1))
    return Y_aligned, (s, R, t), err


def motionbert_frame_error(X_tri: np.ndarray, X_mb: np.ndarray):
    """
    Per-frame similarity-invariant error between triangulated pose and MotionBERT pose.
    """
    _, (s, R, t), err = procrustes_similarity(X_tri, X_mb)
    return (s, R, t), float(err)


# ============================================================
# Focal-length search
# ============================================================

def evaluate_candidate_focals(
    F: np.ndarray,
    x1_all: np.ndarray,
    x2_all: np.ndarray,
    X_mb_all: np.ndarray,
    cx: float,
    cy: float,
    f1: float,
    f2: float,
    use_cheirality: bool = True,
    cheirality_weight: float = 1e3,
):
    """
    x1_all, x2_all: (T, J, 2)
    X_mb_all: (T, J, 3)
    Returns scalar objective and some diagnostics.
    """
    E = essential_from_F(F, f1, f2, cx, cy)
    candidates = decompose_E(E)

    # Fixed first camera at world origin
    K1 = K_from_f(f1, cx, cy)
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])

    best_val = np.inf
    best = None

    for R, t in candidates:

        # keep translation unit length; scale is not observable from E
        P2 = K_from_f(f2, cx, cy) @ np.hstack([R, t.reshape(3, 1)])

        total_mb_err = 0.0
        total_rep_err = 0.0
        total_cheirality = 0


        X_tri_rel_all = []
        X_mb_all_flat = []

        for X1, X2, Xmb in zip(x1_all, x2_all, X_mb_all):
            X_tri = triangulate_points_linear(P1, P2, X1, X2)
            X_tri -= X_tri[0, :]
            X_tri_rel_all.append(X_tri)
            X_mb_all_flat.append(Xmb)

        X_tri_rel_all = np.concatenate(X_tri_rel_all, axis=0)   # (T*J, 3)
        X_mb_all_flat = np.concatenate(X_mb_all_flat, axis=0)

        _, (s, Rg, tg), err = procrustes_similarity(X_tri_rel_all, X_mb_all_flat)

        for X1, X2, Xmb in zip(x1_all, x2_all, X_mb_all):
            X_tri = triangulate_points_linear(P1, P2, X1, X2)

            # reprojection error
            Xh = np.concatenate([X_tri, np.ones((X_tri.shape[0], 1))], axis=1)
            x1_proj = (P1 @ Xh.T).T
            x2_proj = (P2 @ Xh.T).T
            x1_proj = x1_proj[:, :2] / x1_proj[:, 2:3]
            x2_proj = x2_proj[:, :2] / x2_proj[:, 2:3]
            total_rep_err += np.mean(np.sum((x1_proj - X1) ** 2, axis=1))
            total_rep_err += np.mean(np.sum((x2_proj - X2) ** 2, axis=1))
            '''
            weight_mb = 10**10
            (s, _, _), err = motionbert_frame_error(X_tri, Xmb)
            total_mb_err += weight_mb * (1 / s)**2 * err
            '''
            if use_cheirality:
                total_cheirality += cheirality_count(P1, P2, X_tri)

        weight_mb = 1
        total_mb_err = weight_mb * err * (1 / s)**2

        weight_repr = 0
        loss = weight_repr * (1 / ((2 * cx)**2 + (2 * cy)**2)) * total_rep_err + total_mb_err
        #loss = total_mb_err
        if use_cheirality:
            # maximize points in front of cameras => penalize missing points
            missing = x1_all.shape[0] * x1_all.shape[1] - total_cheirality
            loss += cheirality_weight * missing

        if loss < best_val:
            best_val = loss
            best = {
                "f1": f1,
                "f2": f2,
                "R": R,
                "t": t,
                "E": E,
                "loss": loss,
                "rep_err": total_rep_err,
                "mb_err": total_mb_err,
                "cheirality": total_cheirality,
            }

    return best_val, best


def grid_search_focals(
    F: np.ndarray,
    x1_all: np.ndarray,
    x2_all: np.ndarray,
    X_mb_all: np.ndarray,
    cx: float,
    cy: float,
    f1_grid: np.ndarray,
    f2_grid: np.ndarray = None,
):
    """
    If f2_grid is None, assumes f1=f2 and searches one scalar.
    Otherwise searches a 2D grid.
    """

    if f2_grid is None:
        best = None
        for f in f1_grid:
            _, cand = evaluate_candidate_focals(F, x1_all, x2_all, X_mb_all, cx, cy, float(f), float(f))
            if best is None or cand["loss"] < best["loss"]:
                best = cand
        return best

    best = None
    for f1 in f1_grid:
        for f2 in f2_grid:
            _, cand = evaluate_candidate_focals(F, x1_all, x2_all, X_mb_all, cx, cy, float(f1), float(f2))
            if best is None or cand["loss"] < best["loss"]:
                best = cand

    return best


def refine_focals_with_motionbert(
    F: np.ndarray,
    x1_all: np.ndarray,
    x2_all: np.ndarray,
    X_mb_all: np.ndarray,
    cx: float,
    cy: float,
    f1_init: float,
    f2_init: float,
    equal_focals: bool = False,
):
    """
    Continuous refinement of focal lengths using a black-box objective.
    This stays much more stable if started from a grid-search solution.
    """

    if equal_focals:
        def obj(z):
            f = float(np.exp(z[0]))
            val, _ = evaluate_candidate_focals(F, x1_all, x2_all, X_mb_all, cx, cy, f, f)
            return val

        z0 = np.array([np.log(max(f1_init, 1e-3))])
        out = minimize(obj, z0, method="Nelder-Mead", options={"maxiter": 100, "xatol": 1e-3, "fatol": 1e-3})
        f = float(np.exp(out.x[0]))
        _, best = evaluate_candidate_focals(F, x1_all, x2_all, X_mb_all, cx, cy, f, f)
        return best

    def obj(z):
        f1 = float(np.exp(z[0]))
        f2 = float(np.exp(z[1]))
        val, _ = evaluate_candidate_focals(F, x1_all, x2_all, X_mb_all, cx, cy, f1, f2)
        return val

    z0 = np.array([np.log(max(f1_init, 1e-3)), np.log(max(f2_init, 1e-3))])
    out = minimize(obj, z0, method="Nelder-Mead", options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-3})
    f1 = float(np.exp(out.x[0]))
    f2 = float(np.exp(out.x[1]))
    _, best = evaluate_candidate_focals(F, x1_all, x2_all, X_mb_all, cx, cy, f1, f2)
    return best


# ============================================================
# Final reconstruction from chosen cameras
# ============================================================

def reconstruct_sequence(
    x1_all: np.ndarray,
    x2_all: np.ndarray,
    f1: float,
    f2: float,
    cx: float,
    cy: float,
    R: np.ndarray,
    t: np.ndarray,
):
    """
    Triangulate every frame using the selected camera pair.
    Returns X_all: (T,J,3)
    """
    K1 = K_from_f(f1, cx, cy)
    K2 = K_from_f(f2, cx, cy)
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, t.reshape(3, 1)])

    X_all = []
    for X1, X2 in zip(x1_all, x2_all):
        X = triangulate_points_linear(P1, P2, X1, X2)
        X_all.append(X)
    return np.stack(X_all, axis=0)


# ============================================================
# Main pipeline
# ============================================================

def run_pipeline_with_motionbert(
    pts_2d_cam1: np.ndarray,
    pts_2d_cam2: np.ndarray,
    X_mb_all: np.ndarray,
    cx: float,
    cy: float,
    f_min: float = 400.0,
    f_max: float = 3000.0,
    n_grid: int = 40,
    allow_separate_focals: bool = True,
):
    """
    Inputs:
        pts_2d_cam1: (T,J,2)
        pts_2d_cam2: (T,J,2)
        X_mb_all:    (T,J,3) MotionBERT 3D prior, root-centered or otherwise, per frame
    Steps:
        1) estimate F from all correspondences
        2) grid-search focal(s)
        3) refine focal(s)
        4) reconstruct triangulated sequence
    """
    T, J, _ = pts_2d_cam1.shape
    assert pts_2d_cam2.shape == (T, J, 2)
    assert X_mb_all.shape == (T, J, 3)

    # Flatten all correspondences for F estimation
    x1_flat = pts_2d_cam1.reshape(-1, 2)
    x2_flat = pts_2d_cam2.reshape(-1, 2)

    # Optional: remove obvious outliers / NaNs
    mask = np.isfinite(x1_flat).all(axis=1) & np.isfinite(x2_flat).all(axis=1)
    x1_flat = x1_flat[mask]
    x2_flat = x2_flat[mask]

    if x1_flat.shape[0] < 8:
        raise ValueError("Not enough valid correspondences to estimate F")

    # 1) Fundamental matrix
    F, inliers = ransac_F(x1_flat, x2_flat, n_iter=3000, thresh=1.5)
    #print(F)
    #exit()


    # 2) Grid-search focal(s)
    f_grid = np.linspace(f_min, f_max, n_grid)
    best = grid_search_focals(F, pts_2d_cam1, pts_2d_cam2, X_mb_all, cx, cy, f_grid)

    # 3) Refinement
    if allow_separate_focals:
        best = refine_focals_with_motionbert(
            F, pts_2d_cam1, pts_2d_cam2, X_mb_all, cx, cy,
            best["f1"], best["f2"], equal_focals=False
        )
    else:
        best = refine_focals_with_motionbert(
            F, pts_2d_cam1, pts_2d_cam2, X_mb_all, cx, cy,
            best["f1"], best["f2"], equal_focals=True
        )

    # 4) Final reconstruction
    X_rec = reconstruct_sequence(
        pts_2d_cam1, pts_2d_cam2,
        best["f1"], best["f2"], cx, cy,
        best["R"], best["t"],
    )

    result = {
        "F": F,
        "inliers_mask": inliers,
        "f1": best["f1"],
        "f2": best["f2"],
        "R": best["R"],
        "t": best["t"],
        "E": best["E"],
        "loss": best["loss"],
        "reproj_error": best["rep_err"],
        "motionbert_error": best["mb_err"],
        "cheirality": best["cheirality"],
        "X_rec": X_rec,
    }
    return result


# ============================================================
# Example usage
# ============================================================
#if __name__ == "__main__":
    # Replace these with your arrays:
    # pts_2d_cam1: (T,J,2)
    # pts_2d_cam2: (T,J,2)
    # X_mb_all:    (T,J,3)
    # cx, cy: known principal point
#    pass





margins = 80
best_alignment_offset = 46

file_num_1 = 101
file_num_2 = 102
file_num_3 = 107
file_num_4 = 110

file_1 = f"temp_seq/file_{file_num_1}.npy"
file_2 = f"temp_seq/file_{file_num_2}.npy"
base_1 = os.path.splitext(os.path.basename(file_1))[0]
base_2 = os.path.splitext(os.path.basename(file_2))[0]

file_3 = f"temp_seq/file_{file_num_3}.npy"
file_4 = f"temp_seq/file_{file_num_4}.npy"
base_3 = os.path.splitext(os.path.basename(file_3))[0]
base_4 = os.path.splitext(os.path.basename(file_4))[0]

file_number = file_num_1
keypoints_3d = get_kpts(file_number)[margins: -margins]

file_1_len = len(keypoints_3d)

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

skeleton_sums = gaussian_filter1d(skeleton_sums, sigma=2)

for i in range(len(keypoints_3d)):
    keypoints_3d[i] = (np.array(keypoints_3d[i]) / skeleton_sums[i]).tolist()
    for j in range(1, len(keypoints_3d[0])):
        keypoints_3d[i][j] = (np.array(keypoints_3d[i][j]) - np.array(keypoints_3d[i][0])).tolist()
    keypoints_3d[i][0] = [0, 0, 0]

pts_3d_1 = keypoints_3d
skeleton_sums_1 = skeleton_sums

#print(np.max((keypoints_3d * skeleton_sums_1[:, None, None])[:, :, 1]))
#exit()



file_number = file_num_2
keypoints_3d = get_kpts(file_number)

file_2_len = len(keypoints_3d)
diff_len = file_2_len - file_1_len

keypoints_3d = keypoints_3d[best_alignment_offset: len(keypoints_3d) - diff_len + best_alignment_offset]

n = len(h36m_pts)
w0 = np.array([1 / (n)] * n)
skeleton_sums = []
for i in range(len(keypoints_3d)):
    skeleton_sums += [get_ss_weighted(h36m_pts, keypoints_3d[i], w0)]

skeleton_sums = gaussian_filter1d(skeleton_sums, sigma=2)

for i in range(len(keypoints_3d)):
    keypoints_3d[i] = (np.array(keypoints_3d[i]) / skeleton_sums[i]).tolist()
    for j in range(1, len(keypoints_3d[0])):
        keypoints_3d[i][j] = (np.array(keypoints_3d[i][j]) - np.array(keypoints_3d[i][0])).tolist()
    keypoints_3d[i][0] = [0, 0, 0]

pts_3d_2 = keypoints_3d
skeleton_sums_2 = skeleton_sums




#margins = 1

#pts_3d_1 = np.load(file_1, allow_pickle=True)[margins: -margins]
#pts_3d_2 = np.load(file_2, allow_pickle=True)[:]
#pts_3d_1 = np.array(pts_3d_1, dtype=np.float64)
#pts_3d_2 = np.array(pts_3d_2, dtype=np.float64)


#diff_len = len(pts_3d_2) - len(pts_3d_1)

#best_alignment_offset = 9
#pts_3d_2 = pts_3d_2[best_alignment_offset: len(pts_3d_2) - diff_len + best_alignment_offset]

print(len(skeleton_sums_1))
print(len(skeleton_sums_2))

#print(len(pts_3d_1))
#print(len(pts_3d_2))

pts_2d_1 = np.load(f"predictions/{base_1}_2d_synced.npy", allow_pickle=True)
pts_2d_2 = np.load(f"predictions/{base_2}_2d_synced.npy", allow_pickle=True)
pts_2d_1 = np.array(pts_2d_1, dtype=np.float64)
pts_2d_2 = np.array(pts_2d_2, dtype=np.float64)

#print(len(pts_2d_1))
#print(len(pts_2d_2))

#frame_range = range(230, 490)

#ranges = [(230, 250), (350, 370), (450, 470), (550, 570), (650, 670), (750, 770), (850, 870), (950, 970)]
ranges = [(230, 350), (550, 670), (850, 970)]
chunks = [
    pts_2d_1[a:b]
    for (a, b) in ranges
]
pts_2d_1 = np.concatenate(chunks, axis=0)
chunks = [
    pts_2d_2[a:b]
    for (a, b) in ranges
]
pts_2d_2 = np.concatenate(chunks, axis=0)
chunks = [
    pts_3d_1[a:b]
    for (a, b) in ranges
]
pts_3d_1 = np.concatenate(chunks, axis=0)
chunks = [
    pts_3d_2[a:b]
    for (a, b) in ranges
]
pts_3d_2 = np.concatenate(chunks, axis=0)

'''
pts_2d_1 = pts_2d_1[frame_range]
pts_2d_2 = pts_2d_2[frame_range]
pts_3d_1 = pts_3d_1[frame_range]
pts_3d_2 = pts_3d_2[frame_range]
'''

x_all = np.stack([pts_2d_1, pts_2d_2], axis=1)
X_prior_all = pts_3d_2
#skel_sums_prior = skeleton_sums_2

print(x_all.shape)


#W, H = 720, 1280
W, H = 1920, 1080

cx, cy = W / 2, H / 2

#pts_2d_1[:, :, 0] /= (W**2 + H**2)
#pts_2d_1[:, :, 1] /= (W**2 + H**2)
#pts_2d_2[:, :, 0] /= (W**2 + H**2)
#pts_2d_2[:, :, 1] /= (W**2 + H**2)


res = run_pipeline_with_motionbert(
    pts_2d_1,
    pts_2d_2,
    X_prior_all,
    cx,
    cy,
    n_grid=10,
)

print(res['f1'], res['f2'], res['R'], res['t'])

np.save(f'temp_seq/{base_1}_{base_2}_alternated.npy', res['X_rec'])