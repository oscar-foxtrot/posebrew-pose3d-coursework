import json
import cv2

'''
# Read from a JSON file
with open('alpha.json', 'r') as f:
    alpha = json.load(f)
'''


file_number = 727

with open(f'output_file{file_number}/predictions/file_modified_{file_number}.json', 'r') as f:
    rtm = json.load(f)

# print(alpha[0])
#print(alpha[0].keys())

# METRIC TO CHOOSE [THE MAIN PERSON] ID
confs = dict()
for frame in rtm:
    for id, instance in frame['instances'].items():
        try:
            confs[id]['cur_length'] += 1
            confs[id]['value'] = confs[id]['value'] + 1 / confs[id]['cur_length'] * (instance['bbox_score'] - confs[id]['value'])
        except KeyError:
            confs[id] = {'value': instance['bbox_score'], 'cur_length': 1}
times = dict()
for key in confs:
    times[key] = confs[key]['cur_length']
    confs[key]['cur_length'] /= len(rtm)


bboxes_diags = dict()
for frame in rtm:
    for id, instance in frame['instances'].items():
        bbox = instance['bbox'][0]
        bbox_diag = ((bbox[2] - bbox[0])**2 + (bbox[3] - bbox[1])**2)**0.5
        try:
            bboxes_diags[id]['cur_length'] += 1
            bboxes_diags[id]['value'] = bboxes_diags[id]['value'] + 1 / bboxes_diags[id]['cur_length'] * (bbox_diag - bboxes_diags[id]['value'])
        except KeyError:
            bboxes_diags[id] = {'value': bbox_diag, 'cur_length': 1}

max_key = max(bboxes_diags, key=lambda k: bboxes_diags[k]['value'])
max_area = bboxes_diags[max_key]['value']

for key in bboxes_diags:
    bboxes_diags[key]['value'] /= max_area
    bboxes_diags[key]['cur_length'] /= len(rtm)


cap = cv2.VideoCapture(f'neurologist/file_{file_number}.mp4')
if not cap.isOpened():
    raise IOError("Cannot open video file")
# Get video dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x, center_y = frame_width / 2, frame_height / 2
cap.release()

off_center = dict()
max_dist = (center_x**2 + center_y**2)**0.5
for frame in rtm:
    for id, instance in frame['instances'].items():
        bbox = instance['bbox'][0]
        bbox_center_x, bbox_center_y = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        bbox_center_dist_off_center = ((bbox_center_x - center_x)**2 + (bbox_center_y - center_y)**2)**0.5
        try:
            off_center[id]['cur_length'] += 1
            off_center[id]['value'] = off_center[id]['value'] + 1 / off_center[id]['cur_length'] * (bbox_center_dist_off_center - off_center[id]['value'])
        except KeyError:
            off_center[id] = {'value': bbox_center_dist_off_center, 'cur_length': 1}

for key in off_center:
    off_center[key]['value'] = (max_dist - off_center[key]['value']) / max_dist
    off_center[key]['cur_length'] /= len(rtm)


time_in_frame = dict()
for id in confs:
    time_in_frame[id] = confs[id]['cur_length']

for id in confs:
    confs[id] = confs[id]['value']
    bboxes_diags[id] = bboxes_diags[id]['value']
    off_center[id] = off_center[id]['value']

def get_presence_score(weight_time_in_frame, weight_confs, weight_bboxes_diags, weight_off_center,
    time_in_frame, confs, bboxes_diags, off_center):
    scores = dict()
    for id in confs:
        scores[id] = confs[id] * weight_confs + time_in_frame[id] * weight_time_in_frame + \
            bboxes_diags[id] * weight_bboxes_diags + off_center[id] * weight_off_center
    return scores

presence_scores = get_presence_score(1/4, 1/4, 1/4, 1/4,
    time_in_frame, confs, bboxes_diags, off_center)



'''
print(time_in_frame)
print(confs)
print(bboxes_diags)
print(off_center)
print(presence_scores)
'''

id = max(presence_scores, key=presence_scores.get)
# OVERRIDE IF NECESSARY. CHOOSE THE PERSON IN THE VIDEO
# id = '2'

print(times, len(rtm), times['2']/len(rtm)*100)
"""
print(presence_scores)
d = presence_scores

res = ''
for key in d:
    res += f'({varn},{d[key]}) '

print(res)
"""