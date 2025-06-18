
import cv2
import random
import colorsys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
from boxmot import BoostTrack
from pathlib import Path

sys.path.append('./')

# Initialize the tracker
tracker = BoostTrack(
    reid_weights=Path('models/osnet_x1_0_imagenet.pth'),
    device='cpu',
    half=False,
    det_thresh=0.3,
    max_age=100
)


# Initialize random seed for consistent colors
random.seed(100)
id_colors = {}

def get_color(track_id):
    if track_id not in id_colors:
        h = random.random()
        s, v = 1.0, 1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        id_colors[track_id] = (int(b * 255), int(g * 255), int(r * 255))  # BGR for OpenCV
    return id_colors[track_id]

file_number = 718

# Load keypoints
with open(f'../output_file{file_number}/predictions/file_{file_number}.json', 'r') as f:
    kpts = json.load(f)

# Load video
cap = cv2.VideoCapture(f'../neurologist/file_{file_number}.mp4')
if not cap.isOpened():
    raise IOError("Cannot open video file")

# Get video dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'output2_{file_number}.mp4', fourcc, fps, (frame_width, frame_height))

frame_idx = 0
while True:
    ret, img = cap.read()
    if not ret or frame_idx >= len(kpts):
        break

    frame_dets = []
    for instance in kpts[frame_idx]['instances']:
        one_set = []
        one_set += instance['bbox'][0]
        one_set += [instance['bbox_score']]
        frame_dets += [one_set]

    dets = np.hstack([frame_dets, np.zeros((len(frame_dets), 1))]).astype(np.float32)
    tracked_objects = tracker.update(dets, img)

    new_instances = dict()
    for track in tracked_objects:
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])
        color = get_color(track_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f'ID {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        new_instances[track_id] = kpts[frame_idx]['instances'][int(track[7])]

    kpts[frame_idx]['instances'] = new_instances

    out.write(img)
    frame_idx += 1

cap.release()
out.release()
print("Video saved successfully.")

with open(f'../output_file{file_number}/predictions/file_modified_{file_number}.json', 'w', encoding='utf-8') as f:
    json.dump(kpts, f, ensure_ascii=False, indent=4)
