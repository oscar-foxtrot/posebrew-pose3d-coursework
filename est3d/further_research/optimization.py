from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
h36m_pts = [(3, 2), (2, 1), (1, 0), (0, 4), (4, 5), (5, 6), \
    (13, 12), (12, 11), (11, 8), (8, 14), (14, 15), (15, 16), \
    (8, 9), (9, 10), (8, 7), (7, 0)]

file_number = 795
file_path = f'file_{file_number}_2_withoutnorm.npy'
keypoints = np.load(file_path, allow_pickle=True)

# Исходные данные: d(t), d_i(t), t
# Задать начальный w0 (например, равномерный)
n = len(h36m_pts)

#skeleton_sums = []

def get_ss_weighted(list_kpts_tuples, frame_kpts, weights):
    ss = 0
    for i in range(len(list_kpts_tuples)):
        kpt2_coords = frame_kpts[list_kpts_tuples[i][1]]
        kpt1_coords = frame_kpts[list_kpts_tuples[i][0]]
        ss += weights[i] * ((kpt2_coords[2] - kpt1_coords[2])**2 + (kpt2_coords[1] - kpt1_coords[1])**2 + (kpt2_coords[0] - kpt1_coords[0])**2)**0.5
    return ss

"""
for i in range(len(keypoints)):
    skeleton_sums += [get_ss_weighted(h36m_pts, keypoints[i], w0)]

plt.plot(skeleton_sums)
plt.show()
"""

f_vals = []
consts = np.array([1] + [0] * (len(h36m_pts) - 1))
for i in range(len(keypoints)):
    f_vals += [get_ss_weighted(h36m_pts, keypoints[i], consts)]

## REMOVE BELOW
def get_f(frame_kpts):
    ss = 0
    kpt2_coords = frame_kpts[0]
    kpt1_coords = frame_kpts[8]
    ss += ((kpt2_coords[2] - kpt1_coords[2])**2 + (kpt2_coords[1] - kpt1_coords[1])**2 + (kpt2_coords[0] - kpt1_coords[0])**2)**0.5
    return ss

f_vals = []
for i in range(len(keypoints)):
    f_vals += [get_f(keypoints[i])]

## REMOVE ABOVE

#plt.plot(f_vals)
#plt.show()

"""
skeleton_sums = []
for i in range(len(keypoints)):
    skeleton_sums += [get_ss_weighted(h36m_pts, keypoints[i], w0)]
denom = np.array(skeleton_sums, dtype=np.float64)
# вектор размер N: сумма w_i * d_i(t)
plt.plot(f_vals / denom)
plt.show()
"""
"""
w0 = np.array([0] + [1 / (n - 1)] * (n - 1))
def g(w):
    skeleton_sums = []
    for i in range(len(keypoints)):
        skeleton_sums += [get_ss_weighted(h36m_pts, keypoints[i], w)]
    denom = np.array(skeleton_sums, dtype=np.float64)
    # вектор размер N: сумма w_i * d_i(t)
    return f_vals / denom

def objective(w):
    gt = g(w)
    mean_gt = np.mean(gt)
    return np.sum((gt - mean_gt)**2)

t = np.arange(len(keypoints))

def constraint_trend(w):
    gt = g(w)
    mean_gt = np.mean(gt)
    return np.sum((gt - mean_gt)*(t - np.mean(t)))

cons = (
    {'type': 'eq', 'fun': constraint_trend},
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
    #{'type': 'eq', 'fun': lambda w: w[0]}
)

bounds = [(0, None)]*len(w0)

result = minimize(objective, w0, method='SLSQP', constraints=cons, bounds=bounds)

print(result)
print(result['x'])

w0 = result['x']
skeleton_sums = []
for i in range(len(keypoints)):
    skeleton_sums += [get_ss_weighted(h36m_pts, keypoints[i], w0)]
plt.plot(skeleton_sums)
plt.show()

skeleton_sums = []
for i in range(len(keypoints)):
    skeleton_sums += [get_ss_weighted(h36m_pts, keypoints[i], w0)]
denom = np.array(skeleton_sums, dtype=np.float64)
# вектор размер N: сумма w_i * d_i(t)
plt.plot(f_vals / denom)
plt.show()
"""


w0 = '[5.63785130e-18 0.00000000e+00 8.23449842e-18 1.20361940e-18 0.00000000e+00 \
2.18575754e-03 4.55219022e-03 1.81957315e-03 0.00000000e+00 0.00000000e+00 4.84119249e-19 \
0.00000000e+00 3.15435840e-02 1.16216630e-02 5.31926388e-01 4.16350845e-01]'
w0_floats = np.fromstring(w0.strip('[]'), sep=' ')
print(w0_floats)
skeleton_sums = []
for i in range(len(keypoints)):
    skeleton_sums += [get_ss_weighted(h36m_pts, keypoints[i], w0_floats)]
plt.plot(skeleton_sums, label='file_795, позвоночник, без ограничения на тренд', alpha=0.4, color='red')

w0 = '[0.00000000e+00 1.61180566e-18 0.00000000e+00 3.32431708e-18 \
 0.00000000e+00 2.79468643e-03 5.91443004e-03 2.31930418e-03 \
 1.19622153e-18 0.00000000e+00 0.00000000e+00 2.32889488e-18 \
 3.16053928e-02 1.35718091e-02 5.43419936e-01 4.00374441e-01]'
w0_floats = np.fromstring(w0.strip('[]'), sep=' ')
print(w0_floats)
skeleton_sums = []
for i in range(len(keypoints)):
    skeleton_sums += [get_ss_weighted(h36m_pts, keypoints[i], w0_floats)]
plt.plot(skeleton_sums, label='file_795, позвоночник, с ограничением на тренд', alpha=0.4, color='blue')


w0 = '[6.23416249e-19 8.97178353e-01 7.42624278e-17 0.00000000e+00 \
 0.00000000e+00 1.02821647e-01 0.00000000e+00 3.10027611e-17 \
 1.67395394e-16 0.00000000e+00 3.75079742e-17 0.00000000e+00 \
 1.54168128e-16 5.78909750e-17 7.58399420e-18 1.60781761e-16]'
w0_floats = np.fromstring(w0.strip('[]'), sep=' ')
print(w0_floats)
skeleton_sums = []
for i in range(len(keypoints)):
    skeleton_sums += [get_ss_weighted(h36m_pts, keypoints[i], w0_floats)]
plt.plot(np.array(skeleton_sums) * 0.66, label='file_795, голень, без ограничения на тренд', alpha=0.4, color='green')

w0_floats = [1/n] * n
print(w0_floats)
skeleton_sums = []
for i in range(len(keypoints)):
    skeleton_sums += [get_ss_weighted(h36m_pts, keypoints[i], w0_floats)]
plt.plot(np.array(skeleton_sums) * 1.08, label='file_795, сумма всех костей', alpha=0.4, color='orange')

plt.legend()
plt.savefig('Comparison_795.png', dpi=300)
plt.show()
