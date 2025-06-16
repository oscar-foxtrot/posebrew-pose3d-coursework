import json
import numpy as np

# For all files
'''
for file_number in [356, 469, 474, 727, 730, 734, 795, 826, 949, 1081]:

    with open(f'output_file{file_number}/predictions/file_{file_number}.json', 'r') as f:
        rtm = json.load(f)

    res_all = []
    for i in range(len(rtm)):
        res = []
        for j in range(len(rtm[i]['instances'])):
            res += [rtm[i]['instances'][j]['bbox'][0] + [rtm[i]['instances'][j]['bbox_score']]]
        res_all += [res]

    res_all = np.array(res_all, dtype=object)
    np.save(f'output_file{file_number}/predictions/file_{file_number}_bbox.npy', np.array(res_all))
'''


# For an individual file

file_number = 469

with open(f'output_file{file_number}/predictions/file_{file_number}.json', 'r') as f:
    rtm = json.load(f)

res_all = []
for i in range(len(rtm)):
    res = []
    for j in range(len(rtm[i]['instances'])):
        res += [rtm[i]['instances'][j]['bbox'][0] + [rtm[i]['instances'][j]['bbox_score']]]
    res_all += [res]

res_all = np.array(res_all, dtype=object)
np.save(f'output_file{file_number}/predictions/file_{file_number}_bbox.npy', np.array(res_all))