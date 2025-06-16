
# Initialize 3D pose inferencer with MotionBERT (pretrained on Human3.6M)
from mmpose.apis import MMPoseInferencer

# Initialize 3D pose inferencer
inferencer = MMPoseInferencer(
    pose2d='rtmpose-l_8xb512-700e_body8-halpe26-384x288'
    #'rtmpose-m_8xb512-700e_body8-halpe26-256x192',  # explicitly set Halpe26 format
)

file_number = 718

# Run inference with automatic saving
result_generator = inferencer(
    inputs=f'neurologist/file_{file_number}.mp4',
    show=True,
    out_dir=f'output_file{file_number}',
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
