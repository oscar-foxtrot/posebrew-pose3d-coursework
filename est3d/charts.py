import numpy as np
import matplotlib.pyplot as plt

file_number = 469474

keypoints = np.load('COMBINED_469_474.npy', allow_pickle=True)
'''
# Thighs

#keypoints = np.load('COMBINED_469_474.npy', allow_pickle=True)

point_0 = 1
point_1 = 2

diff = keypoints[:, point_0, :] - keypoints[:, point_1, :]
diff = np.array(diff, dtype=np.float32)
dist0 = np.linalg.norm(diff, axis=1)


point_0 = 4
point_1 = 5

diff = keypoints[:, point_0, :] - keypoints[:, point_1, :]
diff = np.array(diff, dtype=np.float32)
dist1 = np.linalg.norm(diff, axis=1)

start = 0
end = 2000

offset = 18
start_offset = 0 if start + offset <= 0 else start + offset
end_offset = end + offset
new_end = start + len(dist1[start_offset: end_offset])

plt.plot(range(len(dist0))[start: end], dist0[start: end], label='Правое бедро')
#plt.plot(range(len(dist1))[start: end], dist1[start: end], label=f'Левое бедро')
plt.plot(range(len(dist1))[start_offset: end_offset], dist1[start: new_end], label=f'Левое бедро, сдвиг на {offset} кадров')
plt.legend()
plt.savefig(f'right_and_left_thigs_offset_{file_number}.png', dpi=300)
plt.show()

plt.plot(range(len(dist0))[start: end], dist0[start: end], label='Правое бедро')
plt.plot(range(len(dist1))[start: end], dist1[start: end], label=f'Левое бедро')
plt.legend()
plt.savefig(f'right_and_left_thigs_{file_number}.png', dpi=300)
plt.show()
'''
'''
# Calves

#keypoints = np.load('D:/Programming/Programs_Py/est3d/file_949.npy', allow_pickle=True)

point_0 = 2
point_1 = 3

diff = keypoints[:, point_0, :] - keypoints[:, point_1, :]
diff = np.array(diff, dtype=np.float32)
dist0 = np.linalg.norm(diff, axis=1)


point_0 = 5
point_1 = 6

diff = keypoints[:, point_0, :] - keypoints[:, point_1, :]
diff = np.array(diff, dtype=np.float32)
dist1 = np.linalg.norm(diff, axis=1)

start = 0
end = 2000

offset = 18
start_offset = 0 if start + offset <= 0 else start + offset
end_offset = end + offset
new_end = start + len(dist1[start_offset: end_offset])

plt.plot(range(len(dist0))[start: end], dist0[start: end], label='Правая голень')
#plt.plot(range(len(dist1))[start: end], dist1[start: end], label=f'Левая голень')
plt.plot(range(len(dist1))[start_offset: end_offset], dist1[start: new_end], label=f'Левая голень, сдвиг на {offset} кадров')
plt.legend()
plt.savefig(f'right_and_left_calves_offset_{file_number}.png', dpi=300)
plt.show()

plt.plot(range(len(dist0))[start: end], dist0[start: end], label='Правая голень')
plt.plot(range(len(dist1))[start: end], dist1[start: end], label=f'Левая голень')
plt.legend()
plt.savefig(f'right_and_left_calves_{file_number}.png', dpi=300)
plt.show()
'''
'''
# Spine

#keypoints = np.load('D:/Programming/Programs_Py/est3d/file_1081.npy', allow_pickle=True)

point_0 = 0
point_1 = 8

diff = keypoints[:, point_0, :] - keypoints[:, point_1, :]
diff = np.array(diff, dtype=np.float32)
dist0 = np.linalg.norm(diff, axis=1)

start = 0
end = 2000

plt.plot(range(len(dist0))[start: end], dist0[start: end], label='Позвоночник')
plt.legend()
plt.savefig(f'spine_{file_number}.png', dpi=300)
plt.show()
'''

# Shoulder lines

#keypoints = np.load('D:/Programming/Programs_Py/est3d/file_1081.npy', allow_pickle=True)
'''
point_0 = 8
point_1 = 14

diff = keypoints[:, point_0, :] - keypoints[:, point_1, :]
diff = np.array(diff, dtype=np.float32)
dist0 = np.linalg.norm(diff, axis=1)


point_0 = 8
point_1 = 11

diff = keypoints[:, point_0, :] - keypoints[:, point_1, :]
diff = np.array(diff, dtype=np.float32)
dist1 = np.linalg.norm(diff, axis=1)

start = 0
end = 2000

offset = 18
start_offset = 0 if start + offset <= 0 else start + offset
end_offset = end + offset
new_end = start + len(dist1[start_offset: end_offset])

plt.plot(range(len(dist0))[start: end], dist0[start: end], label='Правая линия шея-плечо')
#plt.plot(range(len(dist1))[start: end], dist1[start: end], label=f'Левая линия шея-плечо')
plt.plot(range(len(dist1))[start_offset: end_offset], dist1[start: new_end], label=f'Левая линия шея-плечо, сдвиг на {offset} кадров')
plt.legend()
plt.savefig(f'right_and_left_shoulders_offset_{file_number}.png', dpi=300)
plt.show()

plt.plot(range(len(dist0))[start: end], dist0[start: end], label='Правая линия шея-плечо')
plt.plot(range(len(dist1))[start: end], dist1[start: end], label=f'Левая линия шея-плечо')
plt.legend()
plt.savefig(f'right_and_left_shoulders_{file_number}.png', dpi=300)
plt.show()
# Shoulder line

#keypoints = np.load('D:/Programming/Programs_Py/est3d/file_469.npy', allow_pickle=True)

point_0 = 11
point_1 = 14

diff = keypoints[:, point_0, :] - keypoints[:, point_1, :]
diff = np.array(diff, dtype=np.float32)
dist0 = np.linalg.norm(diff, axis=1)

start = 0
end = 2000

plt.plot(range(len(dist0))[start: end], dist0[start: end], label='Линия плеч')
plt.legend()
plt.savefig(f'shoulder_line_{file_number}.png', dpi=300)
plt.show()
'''
'''
# Pelvis lines

#keypoints = np.load('D:/Programming/Programs_Py/est3d/file_949.npy', allow_pickle=True)

point_0 = 0
point_1 = 1

diff = keypoints[:, point_0, :] - keypoints[:, point_1, :]
diff = np.array(diff, dtype=np.float32)
dist0 = np.linalg.norm(diff, axis=1)


point_0 = 0
point_1 = 4

diff = keypoints[:, point_0, :] - keypoints[:, point_1, :]
diff = np.array(diff, dtype=np.float32)
dist1 = np.linalg.norm(diff, axis=1)

start = 0
end = 2000

offset = 18
start_offset = 0 if start + offset <= 0 else start + offset
end_offset = end + offset
new_end = start + len(dist1[start_offset: end_offset])

plt.plot(range(len(dist0))[start: end], dist0[start: end], label='Таз право')
plt.plot(range(len(dist1))[start: end], dist1[start: end], label=f'Таз лево')
#plt.plot(range(len(dist1))[start_offset: end_offset], dist1[start: new_end], label=f'Таз лево, сдвиг на {offset} кадров')
plt.legend()
plt.savefig(f'right_and_left_pelvis_{file_number}.png', dpi=300)
plt.show()

plt.plot(range(len(dist0))[start: end], dist0[start: end], label='Таз право')
#plt.plot(range(len(dist1))[start: end], dist1[start: end], label=f'Таз лево')
plt.plot(range(len(dist1))[start_offset: end_offset], dist1[start: new_end], label=f'Таз лево, сдвиг на {offset} кадров')
plt.legend()
plt.savefig(f'right_and_left_pelvis_offset_{file_number}.png', dpi=300)
plt.show()


# Pelvis line

#keypoints = np.load('D:/Programming/Programs_Py/est3d/file_1081.npy', allow_pickle=True)

point_0 = 1
point_1 = 4

diff = keypoints[:, point_0, :] - keypoints[:, point_1, :]
diff = np.array(diff, dtype=np.float32)
dist0 = np.linalg.norm(diff, axis=1)

start = 0
end = 2000

plt.plot(range(len(dist0))[start: end], dist0[start: end], label='Таз')
plt.legend()
plt.savefig(f'pelvis_line_{file_number}.png', dpi=300)
plt.show()
'''
# Knee angle
'''
#keypoints = np.load('D:/Programming/Programs_Py/est3d/file_795.npy', allow_pickle=True)

point_0 = 1
point_1 = 2

diff = keypoints[:, point_1, :] - keypoints[:, point_0, :]
diff0 = np.array(diff, dtype=np.float32)
dist0 = np.linalg.norm(diff0, axis=1)

point_0 = 2
point_1 = 3

diff = keypoints[:, point_1, :] - keypoints[:, point_0, :]
diff1 = np.array(diff, dtype=np.float32)
dist1 = np.linalg.norm(diff1, axis=1)

crosses_norms = np.linalg.norm(np.cross(diff0, diff1, axis=1), axis=1)
norm_multiples = dist0 * dist1

sin_angles = crosses_norms / norm_multiples
angles = np.arcsin(sin_angles)
angles_degrees = np.degrees(angles)

angles_degrees0 = angles_degrees


point_0 = 4
point_1 = 5

diff = keypoints[:, point_1, :] - keypoints[:, point_0, :]
diff0 = np.array(diff, dtype=np.float32)
dist0 = np.linalg.norm(diff0, axis=1)

point_0 = 5
point_1 = 6

diff = keypoints[:, point_1, :] - keypoints[:, point_0, :]
diff1 = np.array(diff, dtype=np.float32)
dist1 = np.linalg.norm(diff1, axis=1)

crosses_norms = np.linalg.norm(np.cross(diff0, diff1, axis=1), axis=1)
norm_multiples = dist0 * dist1

sin_angles = crosses_norms / norm_multiples
angles = np.arcsin(sin_angles)
angles_degrees = np.degrees(angles)

angles_degrees1 = angles_degrees

start = 0
end = 2000

offset = 18
start_offset = 0 if start + offset <= 0 else start + offset
end_offset = end + offset
new_end = start + len(dist1[start_offset: end_offset])

plt.plot(range(len(angles_degrees0))[start: end], angles_degrees0[start: end], label='Углы, правое колено')
#plt.plot(range(len(angles_degrees1))[start: end], angles_degrees1[start: end], label='Углы, левое колено')
plt.plot(range(len(angles_degrees1))[start_offset: end_offset], angles_degrees1[start: new_end], label=f'Углы, левое колено, сдвиг на {offset} кадров')
plt.legend()
plt.savefig(f'knee_angles_offset_{file_number}.png', dpi=300)
plt.show()

plt.plot(range(len(angles_degrees0))[start: end], angles_degrees0[start: end], label='Углы, правое колено')
plt.plot(range(len(angles_degrees1))[start: end], angles_degrees1[start: end], label='Углы, левое колено')
plt.legend()
plt.savefig(f'knee_angles_{file_number}.png', dpi=300)
plt.show()


# Knee angle

#eypoints = np.load('D:/Programming/Programs_Py/est3d/file_949.npy', allow_pickle=True)

point_0 = 1
point_1 = 2

diff = keypoints[:, point_1, :] - keypoints[:, point_0, :]
diff0 = np.array(diff, dtype=np.float32)
dist0 = np.linalg.norm(diff0, axis=1)

point_0 = 2
point_1 = 3

diff = keypoints[:, point_1, :] - keypoints[:, point_0, :]
diff1 = np.array(diff, dtype=np.float32)
dist1 = np.linalg.norm(diff1, axis=1)

crosses_norms = np.linalg.norm(np.cross(diff0, diff1, axis=1), axis=1)
norm_multiples = dist0 * dist1

sin_angles = crosses_norms / norm_multiples
angles = np.arcsin(sin_angles)
angles_degrees = np.degrees(angles)

angles_degrees0 = angles_degrees


point_0 = 4
point_1 = 5

diff = keypoints[:, point_1, :] - keypoints[:, point_0, :]
diff0 = np.array(diff, dtype=np.float32)
dist0 = np.linalg.norm(diff0, axis=1)

point_0 = 5
point_1 = 6

diff = keypoints[:, point_1, :] - keypoints[:, point_0, :]
diff1 = np.array(diff, dtype=np.float32)
dist1 = np.linalg.norm(diff1, axis=1)

crosses_norms = np.linalg.norm(np.cross(diff0, diff1, axis=1), axis=1)
norm_multiples = dist0 * dist1

sin_angles = crosses_norms / norm_multiples
angles = np.arcsin(sin_angles)
angles_degrees = np.degrees(angles)

angles_degrees1 = angles_degrees

start = 0
end = 2000

offset = 18
start_offset = 0 if start + offset <= 0 else start + offset
end_offset = end + offset
new_end = start + len(dist1[start_offset: end_offset])

plt.plot(range(len(angles_degrees0))[start: end], 180 - angles_degrees0[start: end], label='Углы, правое колено')
plt.plot(range(len(angles_degrees1))[start: end], 180 - angles_degrees1[start: end], label='Углы, левое колено')
#plt.plot(range(len(angles_degrees1))[start_offset: end_offset], 180 - angles_degrees1[start: new_end], label=f'Углы, левое колено, сдвиг на {offset} кадров')
plt.legend()
plt.savefig(f'inv_knee_angles_{file_number}.png', dpi=300)
plt.show()

plt.plot(range(len(angles_degrees0))[start: end], 180 - angles_degrees0[start: end], label='Углы, правое колено')
#plt.plot(range(len(angles_degrees1))[start: end], 180 - angles_degrees1[start: end], label='Углы, левое колено')
plt.plot(range(len(angles_degrees1))[start_offset: end_offset], 180 - angles_degrees1[start: new_end], label=f'Углы, левое колено, сдвиг на {offset} кадров')
plt.legend()
plt.savefig(f'inv_knee_angles_offset_{file_number}.png', dpi=300)
plt.show()
'''