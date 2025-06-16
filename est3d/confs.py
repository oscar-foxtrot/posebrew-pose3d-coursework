import json
import cv2
import numpy as np

'''
# Read from a JSON file
with open('alpha.json', 'r') as f:
    alpha = json.load(f)
'''

files_1 = [356, 474, 718, 795, 949, 1081]
files_2 = [469, 727, 730, 734, 826]

files = [356, 469, 474, 718, 727, 730, 734, 795, 826, 949, 1081]

i = 0
for file_number in files:
    i += 1
    indices = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    with open(f'output_file{file_number}/predictions/file_{file_number}_toalpha_0.json', 'r') as f:
        rtm = json.load(f)

    means = []

    if file_number in files_1:
        for frame in rtm[:len(rtm) // 2]:
            confs = np.array(frame['keypoints'][2::3])[indices]
            means.append(np.mean(confs))
    else:
        for frame in rtm[len(rtm) // 2:]:
                confs = np.array(frame['keypoints'][2::3])[indices]
                means.append(np.mean(confs))

    """
    print(len(rtm))
    print(sum(mean > 0.5 for mean in means) / len(rtm) * 100)
    print(sum(mean > 0.75 for mean in means) / len(rtm) * 100)
    print(sum(mean > 0.8 for mean in means) / len(rtm) * 100)
    print(sum(mean > 0.85 for mean in means) / len(rtm) * 100)
    """

    total_frames = len(rtm[:len(rtm) // 2]) if file_number in files_1 else len(rtm[len(rtm) // 2:])
    p05 = sum(mean > 0.7 for mean in means) / total_frames * 100
    p075 = sum(mean > 0.75 for mean in means) / total_frames * 100
    p08 = sum(mean > 0.8 for mean in means) / total_frames * 100
    p085 = sum(mean > 0.85 for mean in means) / total_frames * 100

    print(f"{i} & {total_frames} & {p05:.2f}\\% & {p075:.2f}\\% & {p08:.2f}\\% & {p085:.2f}\\% \\\\")
