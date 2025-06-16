# Initialize 3D pose inferencer with MotionBERT (pretrained on Human3.6M)
from mmpose.apis import MMPoseInferencer

# Initialize 3D pose inferencer
inferencer = MMPoseInferencer(
    pose3d="human3d"
    #'rtmpose-m_8xb512-700e_body8-halpe26-256x192',  # explicitly set Halpe26 format
)

# Run inference with automatic saving
result_generator = inferencer(
    inputs='vid0.mp4',
    show=True,
    out_dir='output0_mmpose_30fps',
    return_vis=True,
    save_pred=True  # <-- This tells it to save predictions automatically
)

while True:
    try:
        result = next(result_generator)
        print(result.keys())  # Do something with each result (e.g., process or save)
    except StopIteration:
        # If we reach the end of the generator, stop the loop
        print("No more results.")
        break  # should include 'keypoints', 'frame_id', etc.

'''
from mmpose.apis import MMPoseInferencer

img_path = 'img.jpg'

inferencer = MMPoseInferencer('human')

result_generator = inferencer(img_path, show=True)
result = next(result_generator)
'''