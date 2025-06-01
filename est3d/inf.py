# Get 2D keypoints from a video
from mmpose.apis import MMPoseInferencer

# Initialize 3D pose inferencer
inferencer = MMPoseInferencer(
    pose2d='rtmpose-l_8xb512-700e_body8-halpe26-384x288'
)

input_file_path = 'input_file.mp4'
output_dir_path = 'output_dir'

# Run inference with automatic saving
result_generator = inferencer(
    inputs=input_file_path,
    show=True,
    out_dir=output_dir_path,
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
