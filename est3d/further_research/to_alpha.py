#Obtain multiple sequences of 2D keypoints, shifted by a stride

import json

'''
# Read from a JSON file
with open('alpha.json', 'r') as f:
    alpha = json.load(f)
'''

input_file_path = 'input_file.json'
with open(input_file_path, 'r') as f:
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

output_file_template = 'output_file'

with open(f'{output_file_template}_toalpha_0.json', 'w') as f:
    json.dump(res, f, indent=4)

with open(f'{output_file_template}_toalpha_1.json', 'w') as f:
    json.dump(res1, f, indent=4)

with open(f'{output_file_template}_toalpha_2.json', 'w') as f:
    json.dump(res2, f, indent=4)
