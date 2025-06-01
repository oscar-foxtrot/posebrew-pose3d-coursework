import numpy as np
import matplotlib.pyplot as plt
from apply_kernel import get_kpts

'''
point_0 = 11
point_1 = 14

diff = keypoints[:, point_0, :] - keypoints[:, point_1, :]
diff = np.array(diff, dtype=np.float32)
dist0 = np.linalg.norm(diff, axis=1)

start = 0
end = 2000

plt.plot(range(len(dist0))[start: end], dist0[start: end], label='Линия плеч')

import json

'''
'''
with open('output_file469/predictions/file_469_toalpha_0.json', 'r') as f:
    kpts = json.load(f)

print(kpts[0]['keypoints'])
'''
'''

with open('output_file469/predictions/file_469.json', 'r') as f:
    kpts = json.load(f)

print(kpts[0]['instances'][0]['keypoints'])


plt.legend()
# plt.savefig('shoulder_line_949.png', dpi=300)
plt.show()
'''

def get_ss(list_kpts_tuples, frame_kpts):
    ss = 0
    for i in range(len(list_kpts_tuples)):
        kpt2_coords = frame_kpts[list_kpts_tuples[i][1]]
        kpt1_coords = frame_kpts[list_kpts_tuples[i][0]]
        ss += ((kpt2_coords[2] - kpt1_coords[2])**2 + (kpt2_coords[1] - kpt1_coords[1])**2 + (kpt2_coords[0] - kpt1_coords[0])**2)**0.5
    return ss

keypoints = get_kpts()
keypoints_3d = keypoints

h36m_pts = [(3,2), (2,1), (1, 0), (0, 4), (4, 5), (5, 6), \
    (13, 12), (12, 11), (11, 8), (8, 14), (14, 15), (15, 16), \
    (8, 9), (9, 10), (8, 7), (7, 0)]
skeleton_sums = []
for i in range(len(keypoints_3d)):
    skeleton_sums += [get_ss(h36m_pts, keypoints_3d[i])]

chosen_pts = [(1, 4)]
chosen_sums = []
for i in range(len(keypoints_3d)):
    chosen_sums += [get_ss(chosen_pts, keypoints_3d[i])]
'''
plt.plot(range(len(skeleton_sums)), np.array(chosen_sums) * 5.2, label=f'Нормализация ТОЛЬКО на сумму 0-8 и 1-4\n(домножено на некоторую константу (5.2) для\nсравнения с нормализацией на все кости)')
plt.plot(range(len(skeleton_sums)), np.array(skeleton_sums) , label=f'Нормализация на сумму всех костей')
'''
#plt.plot(range(len(skeleton_sums)), np.array(chosen_sums) / np.array(skeleton_sums) * 0.5, label=f'Отношение {chosen_pts[0][0]}-{chosen_pts[0][1]} к усреднению (нормализации)\n(умножено на 0.5)')

chosen_pts = [(2, 1)]
chosen_sums = []
for i in range(len(keypoints_3d)):
    chosen_sums += [get_ss(chosen_pts, keypoints_3d[i])]
plt.plot(range(len(skeleton_sums)), np.array(chosen_sums) / np.array(skeleton_sums), label=f'Отношение {chosen_pts[0][0]}-{chosen_pts[0][1]} к усреднению (нормализации)')

chosen_pts = [(5, 4)]
chosen_sums = []
for i in range(len(keypoints_3d)):
    chosen_sums += [get_ss(chosen_pts, keypoints_3d[i])]
plt.plot(range(len(skeleton_sums)), np.array(chosen_sums) / np.array(skeleton_sums), label=f'Отношение {chosen_pts[0][0]}-{chosen_pts[0][1]} к усреднению (нормализации)')
plt.legend()

plt.savefig('chart_469.png', dpi=300)


plt.show()
