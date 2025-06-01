#Obtain multiple sequences of 2D keypoints, shifted by a stride

import json

'''
# Read from a JSON file
with open('alpha.json', 'r') as f:
    alpha = json.load(f)
'''


with open('output_test232_469/predictions/file_469.json', 'r') as f:
    rtm = json.load(f)

# print(alpha[0])
#print(alpha[0].keys())

m = 0
res = []
for i in range(len(rtm)):
    a = dict()
    a['image_id'] = f"{rtm[i]['frame_id']}"
    a['category_id'] = 0
    a['keypoints'] = []
    for k in range(26):
        a['keypoints'] += rtm[i]['instances'][m]['keypoints'][k]
        a['keypoints'] += [rtm[i]['instances'][m]['keypoint_scores'][k]]
    a['box'] = rtm[i]['instances'][m]['bbox'][0]  # ind
    a['idx'] = [0.0]
    res += [a]


res1 = res[243 // 3:]
res2 = res[243 // 3 * 2:]

with open('output_test232_469/predictions/file_469_toalpha_0.json', 'w') as f:
    json.dump(res, f, indent=4)

with open('output_test232_469/predictions/file_469_toalpha_1.json', 'w') as f:
    json.dump(res1, f, indent=4)

with open('output_test232_469/predictions/file_469_toalpha_2.json', 'w') as f:
    json.dump(res2, f, indent=4)
