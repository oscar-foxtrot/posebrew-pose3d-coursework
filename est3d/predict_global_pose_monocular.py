import os
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from apply_kernel import get_kpts

#os.environ["OMP_NUM_THREADS"] = "3"

def halpe2h36m(x):
    T, V, C = x.shape
    y = np.zeros([T,17,C])
    y[:,0,:] = x[:,19,:]
    y[:,1,:] = x[:,12,:]
    y[:,2,:] = x[:,14,:]
    y[:,3,:] = x[:,16,:]
    y[:,4,:] = x[:,11,:]
    y[:,5,:] = x[:,13,:]
    y[:,6,:] = x[:,15,:]
    y[:,7,:] = (x[:,18,:] + x[:,19,:]) * 0.5
    y[:,8,:] = x[:,18,:]
    y[:,9,:] = x[:,0,:]
    y[:,10,:] = x[:,17,:]
    y[:,11,:] = x[:,5,:]
    y[:,12,:] = x[:,7,:]
    y[:,13,:] = x[:,9,:]
    y[:,14,:] = x[:,6,:]
    y[:,15,:] = x[:,8,:]
    y[:,16,:] = x[:,10,:]
    return y



def estimate_global_f(
    keypoints_2d,
    keypoints_3d,
    skeleton_sums,
    varlambda,
    cx,
    cy,
    eps=1e-12
):
    A = 0.0
    B = 0.0

    T = len(keypoints_3d)

    for t in range(T):
        pose = np.asarray(keypoints_3d[t])
        u = keypoints_2d[t][:, 0]
        v = keypoints_2d[t][:, 1]

        
        S = 1 / skeleton_sums[t]
        #S = 1 / (skeleton_sums[t] / np.mean(skeleton_sums[t])) * 36

        # root 2D (you used this in your model)
        u0 = u[0]
        v0 = v[0]

        X = pose[:, 0]
        Y = pose[:, 1]
        Z = pose[:, 2]

        denom = Z + varlambda * S

        valid = np.abs(denom) > 0 #eps
        if np.sum(valid) < 2:
            continue

        a = X[valid] / denom[valid]
        b = Y[valid] / denom[valid]

        alpha = cx + (varlambda * S / denom[valid]) * (u0 - cx)
        beta  = cy + (varlambda * S / denom[valid]) * (v0 - cy)

        u_v = u[valid]
        v_v = v[valid]

        A += np.sum(a * a + b * b)
        B += np.sum((u_v - alpha) * a + (v_v - beta) * b)

    #return B / (A + 1e-8)
    return B / A

file_number = 101

with open(f"./mm_output/output_file_{file_number}/predictions/file_{file_number}_toalpha_0.json", "r") as f:
    data = json.load(f)

#W, H = 720, 1280
W, H = 1920, 1080
cx, cy = W / 2, H / 2

keypoints_3d = get_kpts(file_number)

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

keypoints_2d = np.array([np.array(data[frame_number]["keypoints"]).reshape(-1, 3) for frame_number in range(len(data))])
confs = halpe2h36m(keypoints_2d[:, :, 2:3])
keypoints_2d = halpe2h36m(keypoints_2d[:, :, 0:2])

confs = confs.squeeze(axis=-1)


frame_signal = (confs[:, 3] + confs[:, 6]) / 2
frame_signal = gaussian_filter1d(frame_signal, sigma=15)

print(len(keypoints_3d))
print(len(keypoints_2d))

# Feature engineering BEGIN

LEFT_FOOT = 6
RIGHT_FOOT = 3
PELVIS = 0


T, J, _ = keypoints_3d.shape

left_features = []
right_features = []

# right leg
upper = keypoints_3d[:, 1] - keypoints_3d[:, 2]
lower = keypoints_3d[:, 2] - keypoints_3d[:, 3]

npdot = np.sum(upper * lower, axis=1)
norm_upper = np.linalg.norm(upper, axis=1)
norm_lower = np.linalg.norm(lower, axis=1)

cosine_left = npdot / (norm_upper * norm_lower) - (1 - npdot / (norm_upper * norm_lower)) * 0.95

# left leg
upper = keypoints_3d[:, 4] - keypoints_3d[:, 5]
lower = keypoints_3d[:, 5] - keypoints_3d[:, 6]

npdot = np.sum(upper * lower, axis=1)
norm_upper = np.linalg.norm(upper, axis=1)
norm_lower = np.linalg.norm(lower, axis=1)

cosine_right = npdot / (norm_upper * norm_lower) - (1 - npdot / (norm_upper * norm_lower)) * 0.95 

l_acc = keypoints_3d[2:, LEFT_FOOT] - 2 * keypoints_3d[1:-1, LEFT_FOOT] + keypoints_3d[:-2, LEFT_FOOT]
l_acc_norm = np.linalg.norm(l_acc, axis=1)

r_acc = keypoints_3d[2:, RIGHT_FOOT] - 2 * keypoints_3d[1:-1, RIGHT_FOOT] + keypoints_3d[:-2, RIGHT_FOOT]
r_acc_norm = np.linalg.norm(r_acc, axis=1)

for t in range(1, T - 1):
    frame = keypoints_3d[t]

    pelvis = frame[PELVIS]

    lfoot = frame[LEFT_FOOT]
    rfoot = frame[RIGHT_FOOT]

    # foot heights (relative to pelvis)
    l_height = lfoot[1] - pelvis[1]
    r_height = rfoot[1] - pelvis[1]

    '''
    if t == 0:
        l_vel = 0.0
        r_vel = 0.0
    '''
    #else:
    prev = keypoints_3d[t - 1]

    l_vel = np.linalg.norm(lfoot - prev[LEFT_FOOT])
    r_vel = np.linalg.norm(rfoot - prev[RIGHT_FOOT])

    l_ankle = frame[5]
    r_ankle = frame[2]
    l_ankle_vel = np.linalg.norm(l_ankle - prev[5])
    r_ankle_vel = np.linalg.norm(r_ankle - prev[2])


    l_arm = frame[13]
    r_arm = frame[16]
    l_arm_vel = np.linalg.norm(l_arm - prev[13])
    r_arm_vel = np.linalg.norm(r_arm - prev[16])
    
    
    left_features.append([
        l_height,
        #r_height,
        l_vel,
        #r_vel,
        cosine_left[t],
        #cosine_right[t],
        l_acc_norm[t - 1],
        #r_acc_norm[t - 1],
        l_ankle_vel,
        #r_ankle_vel,
        #l_arm_vel,
        #r_arm_vel,
    ])
    
    right_features.append([
        #l_height,
        r_height,
        #l_vel,
        r_vel,
        #cosine_left[t],
        cosine_right[t],
        #l_acc_norm[t - 1],
        r_acc_norm[t - 1],
        #l_ankle_vel,
        r_ankle_vel,
        #l_arm_vel,
        #r_arm_vel,
    ])
    

left_features = np.array(left_features)
right_features = np.array(right_features)
left_features = (left_features - left_features.mean(axis=0)) / (left_features.std(axis=0) + 1e-8)
right_features = (right_features - right_features.mean(axis=0)) / (right_features.std(axis=0) + 1e-8)

# Feature engineering END

energy = (
    left_features[:, 1] +
    right_features[:, 1] +
    0.5 * np.abs(left_features[:, 2] - right_features[:, 2]) +
    0.5 * np.abs(left_features[:, 0] - right_features[:, 0]) +
    gaussian_filter1d(np.abs(left_features[:, 3]), sigma=30) +
    gaussian_filter1d(np.abs(right_features[:, 3]), sigma=30)
)

energy = gaussian_filter1d(energy, sigma=40)

win=50
scores = []
for w in range(len(energy) - win + 1):
    scores.append(np.mean(energy[w:w+win]) + np.std(energy[w:w+win]))

frames = []
for w, s in enumerate(scores):
    if s > 2.5:  # Threshold
        frames.append(w)

# Cluster the selected frames only

clust_num = 2

gmm = GaussianMixture(
    n_components=2,
    covariance_type="full",
    random_state=0,
    n_init=5
)

left_features = left_features[frames]
right_features = right_features[frames]

left_labels = gmm.fit_predict(left_features)
left_probs = gmm.predict_proba(left_features)

right_labels = gmm.fit_predict(right_features)
right_probs = gmm.predict_proba(right_features)

left_cluster_scores = []
right_cluster_scores = []

for k in range(clust_num):
    idx = left_labels == k
    sub = left_features[idx]
    vel = sub[:, 1]
    a_vel = sub[:, 4]
    score = np.mean(vel**2 + a_vel**2)
    left_cluster_scores.append(score)
left_contact_cluster = np.argmin(left_cluster_scores)

for k in range(clust_num):
    idx = right_labels == k
    sub = right_features[idx]
    vel = sub[:, 1]
    a_vel = sub[:, 4]
    score = np.mean(vel**2 + a_vel**2)
    right_cluster_scores.append(score)
right_contact_cluster = np.argmin(right_cluster_scores)

#w_left = left_probs[:, left_contact_cluster]
#w_left = w_left / (np.sum(w_left) + 1e-8)
#w_right = right_probs[:, right_contact_cluster]
#w_right = w_right / (np.sum(w_right) + 1e-8)


left_contact = (left_labels == left_contact_cluster).astype(np.int32)
right_contact = (right_labels == right_contact_cluster).astype(np.int32)


# EXTRACT CONTINUOUS INTERVALS OF TIME (from frames)

segments = []

start = frames[0]
prev = frames[0]

for f in frames[1:]:
    if f == prev + 1:
        prev = f
    else:
        segments.append((start, prev))
        start = f
        prev = f

segments.append((start, prev))


temp = 4.0
eps = 1e-3

right_probs = np.power(right_probs, 1.0 / temp)
right_probs /= right_probs.sum(axis=1, keepdims=True)

left_probs = np.power(left_probs, 1.0 / temp)
left_probs /= left_probs.sum(axis=1, keepdims=True)

right_probs = np.clip(right_probs, eps, 1.0)
right_probs /= right_probs.sum(axis=1, keepdims=True)

left_probs = np.clip(left_probs, eps, 1.0)
left_probs /= left_probs.sum(axis=1, keepdims=True)


left_contact_prob = left_probs[:, left_contact_cluster]
left_swing_prob = 1.0 - left_contact_prob

right_contact_prob = right_probs[:, right_contact_cluster]
right_swing_prob = 1.0 - right_contact_prob

#############################
fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(
    right_swing_prob,
    color='0.2',
    linewidth=1.2,
    alpha=0.7,
)
axs[0, 1].plot(
    left_swing_prob,
    color='0.2',
    linewidth=1.2,
    alpha=0.7,
)

frame_to_idx = {f: i for i, f in enumerate(frames)}

left_states_all = np.full(len(frames), -1)
right_states_all = np.full(len(frames), -1)

log_trans = np.log(np.array([
    [0.95, 0.05],
    [0.08, 0.92]
]))

for s, e in segments:

    idxs = [frame_to_idx[f] for f in range(s, e + 1)]
    
    lc = left_contact_prob[idxs]
    ls = left_swing_prob[idxs]

    log_emit = np.log(np.stack([lc, ls], axis=1) + 1e-8)

    T = len(log_emit)

    dp = np.zeros((T, 2))
    ptr = np.zeros((T, 2), dtype=int)

    dp[0] = log_emit[0]

    for t in range(1, T):
        for j in range(2):
            vals = dp[t-1] + log_trans[:, j]
            ptr[t, j] = np.argmax(vals)
            dp[t, j] = np.max(vals) + log_emit[t, j]

    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(dp[-1])

    for t in range(T-2, -1, -1):
        states[t] = ptr[t+1, states[t+1]]

    left_states_all[idxs] = states



    rc = right_contact_prob[idxs]
    rs = right_swing_prob[idxs]

    log_emit = np.log(np.stack([rc, rs], axis=1) + 1e-8)

    T = len(log_emit)

    dp = np.zeros((T, 2))
    ptr = np.zeros((T, 2), dtype=int)

    dp[0] = log_emit[0]

    for t in range(1, T):
        for j in range(2):
            vals = dp[t-1] + log_trans[:, j]
            ptr[t, j] = np.argmax(vals)
            dp[t, j] = np.max(vals) + log_emit[t, j]

    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(dp[-1])

    for t in range(T-2, -1, -1):
        states[t] = ptr[t+1, states[t+1]]

    right_states_all[idxs] = states

axs[1, 0].plot(right_states_all, drawstyle="steps-post", color='black', linewidth=2.0)
axs[1, 1].plot(left_states_all, drawstyle="steps-post", color='black', linewidth=2.0)

axs[0, 0].set_ylabel("Оценка вероятности\nфазы маха")
axs[1, 0].set_ylabel("Состояние")

for ax in axs[1, :]:
    #ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])

for ax in axs[1, :]:
    ax.set_xlabel("Кадр")

axs[0, 0].text(0.01, 0.92, "(a)", transform=axs[0, 0].transAxes,
               fontsize=10, va="top")

axs[0, 1].text(0.01, 0.92, "(b)", transform=axs[0, 1].transAxes,
               fontsize=10, va="top")

axs[1, 0].text(0.01, 0.92, "(c)", transform=axs[1, 0].transAxes,
               fontsize=10, va="top")

axs[1, 1].text(0.01, 0.92, "(d)", transform=axs[1, 1].transAxes,
               fontsize=10, va="top")

#plt.savefig("figure_107.pdf", bbox_inches="tight")
plt.show()
#exit()




# LEFT FOOT
contact_bin = left_states_all.astype(np.int32)
diff_left = contact_bin[1:] - contact_bin[:-1]
left_heel_strikes = np.where(diff_left == -1)[0] + 1
left_toe_offs = np.where(diff_left == 1)[0] + 1

# RIGHT FOOT
contact_bin = right_states_all.astype(np.int32)
diff_right = contact_bin[1:] - contact_bin[:-1]
right_heel_strikes = np.where(diff_right == -1)[0] + 1
right_toe_offs = np.where(diff_right == 1)[0] + 1


left_stance = []
for hs in left_heel_strikes:
    to_candidates = left_toe_offs[left_toe_offs > hs]
    if len(to_candidates) == 0:
        continue
    to = to_candidates[0]
    left_stance.append((hs, to))

right_stance = []
for hs in right_heel_strikes:
    to_candidates = right_toe_offs[right_toe_offs > hs]
    if len(to_candidates) == 0:
        continue
    to = to_candidates[0]
    right_stance.append((hs, to))



left_stance_full = [
    (frames[s], frames[e]) for (s, e) in left_stance
]

right_stance_full = [
    (frames[s], frames[e]) for (s, e) in right_stance
]

#varlambda = 0.8

losses = []
lambdas = np.geomspace(0.1, 5, 200)

# IMPORTANT: use FULL-FRAME stance intervals
L_STANCE = left_stance_full
R_STANCE = right_stance_full

for varlambda in lambdas:

    f = estimate_global_f(
        keypoints_2d,
        keypoints_3d,
        skeleton_sums,
        varlambda,
        cx,
        cy,
        eps=1e-8
    )

    X_world = np.zeros_like(keypoints_3d)

    # reconstruct world coords
    for t in range(len(keypoints_3d)):

        Z0 = varlambda * 1.0 / skeleton_sums[t]
        u0, v0 = keypoints_2d[t][0]

        X0 = (u0 - cx) / f * Z0
        Y0 = (v0 - cy) / f * Z0

        X_world[t] = keypoints_3d[t] + np.array([X0, Y0, Z0])

    loss = 0.0
    cutoff = 10

    # LEFT FOOT STANCE LOSS
    for (t_s, t_e) in L_STANCE[:cutoff]:

        if t_e <= t_s:
            continue

        mid = (t_s + t_e) // 2

        a = X_world[t_e, LEFT_FOOT] - X_world[t_s, LEFT_FOOT]
        b = X_world[mid, LEFT_FOOT] - X_world[t_s, LEFT_FOOT]
        c = X_world[t_e, LEFT_FOOT] - X_world[mid, LEFT_FOOT]

        loss += np.dot(a, a) + np.dot(b, b) + np.dot(c, c)

    # RIGHT FOOT STANCE LOSS
    for (t_s, t_e) in R_STANCE[:cutoff]:

        if t_e <= t_s:
            continue

        mid = (t_s + t_e) // 2

        a = X_world[t_e, RIGHT_FOOT] - X_world[t_s, RIGHT_FOOT]
        b = X_world[mid, RIGHT_FOOT] - X_world[t_s, RIGHT_FOOT]
        c = X_world[t_e, RIGHT_FOOT] - X_world[mid, RIGHT_FOOT]

        loss += np.dot(a, a) + np.dot(b, b) + np.dot(c, c)

    print(f"λ={varlambda:.3f}, loss={loss:.6f}")
    losses.append(loss)

best_lambda = lambdas[np.argmin(losses)]

print("BEST λ:", best_lambda)

plt.plot(lambdas, losses)
plt.xscale("log")
plt.show()

#np.save(f"temp_seq/file_{file_number}_markov.npy", keypoints_3d)