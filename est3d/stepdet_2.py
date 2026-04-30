import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from apply_kernel import get_kpts
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


file_number = 474
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



LEFT_FOOT = 6
RIGHT_FOOT = 3
PELVIS = 0


T, J, _ = keypoints_3d.shape

features = []

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
    
    features.append([
        l_height,
        r_height,
        l_vel,
        r_vel,
        cosine_left[t],
        cosine_right[t],
        l_acc_norm[t - 1],
        r_acc_norm[t - 1],
        l_ankle_vel,
        r_ankle_vel,
        #l_arm_vel,
        #r_arm_vel,
    ])
    

features = np.array(features)
features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

clust_num = 2

gmm = GaussianMixture(
    n_components=2,
    covariance_type="full",
    random_state=0,
    n_init=5
)

labels = gmm.fit_predict(features)

probs = gmm.predict_proba(features)

cluster_scores = []

for k in range(clust_num):
    idx = labels == k
    score = np.mean(features[idx, 0:2]) + np.mean(features[idx, 2:4])
    cluster_scores.append(score)

contact_cluster = np.argmin(cluster_scores)
contact_signal = (labels == contact_cluster).astype(float)

#contact_signal = gaussian_filter1d(contact_signal, sigma=2)


plt.figure(figsize=(12, 4))

plt.scatter(
    np.arange(len(labels)),
    labels,
    c=labels,
    cmap="viridis",
    s=5
)


upper = keypoints_3d[:, 1] - keypoints_3d[:, 2]
lower = keypoints_3d[:, 2] - keypoints_3d[:, 3]

npdot = np.sum(upper * lower, axis=1)
norm_upper = np.linalg.norm(upper, axis=1)
norm_lower = np.linalg.norm(lower, axis=1)

cosine = npdot / (norm_upper * norm_lower) - (1 - npdot / (norm_upper * norm_lower)) * 0.95

plt.plot(cosine)
plt.show()


switch_rate = np.mean(labels[1:] != labels[:-1])
score = silhouette_score(features, labels)
print(score, switch_rate)


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

n_components = len(features[0])
pca = PCA(n_components=4)
features_pca = pca.fit_transform(features)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=0,
    init="pca"
)

features_2d = tsne.fit_transform(features_pca)

plt.figure(figsize=(6, 6))

plt.scatter(
    features_2d[:, 0],
    features_2d[:, 1],
    c=labels,
    cmap="viridis",
    s=5
)

plt.title("t-SNE of gait features (colored by KMeans clusters)")
plt.show()


pca = PCA()
pca.fit(features)
explained = pca.explained_variance_ratio_
plt.plot(explained, marker='o')
plt.title("PCA Explained Variance (Scree Plot)")
plt.xlabel("Component")
plt.ylabel("Variance ratio")
plt.grid()
plt.show()



contact_cluster = np.argmin(cluster_scores)
plt.figure(figsize=(12, 4))

plt.plot(probs[:, contact_cluster], label="contact probability")
plt.plot(cosine, alpha=0.5, label="cosine (sanity)")
plt.legend()
plt.show()


plt.plot(labels)
plt.show()