import json

'''
# Read from a JSON file
with open('alpha.json', 'r') as f:
    alpha = json.load(f)
'''


with open('vid2.json', 'r') as f:
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
        '''
        a['keypoints'] += [rtm[i]['instances'][m]['keypoints'][k][0] / (bbox[2] - bbox[0])]
        a['keypoints'] += [rtm[i]['instances'][m]['keypoints'][k][1] / (bbox[3] - bbox[1])]
        '''
        a['keypoints'] += [(rtm[i]['instances'][m]['keypoints'][k][0] - bbox_center[0]) / bbox_norm]
        a['keypoints'] += [(rtm[i]['instances'][m]['keypoints'][k][1] - bbox[1]) / bbox_norm]
        a['keypoints'] += [rtm[i]['instances'][m]['keypoint_scores'][k]]
    a['box'] = bbox  # ind
    a['idx'] = [0.0]
    res += [a]


print(res[200])


with open('vid2_toalpha_normalized_bottom.json', 'w') as f:
    json.dump(res, f, indent=4)
