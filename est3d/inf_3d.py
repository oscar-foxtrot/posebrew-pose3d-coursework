# Initialize 3D pose inferencer with MotionBERT (pretrained on Human3.6M)
# Get a glimpse of the results achieved with MMPose's implementation

from mmpose.apis import MMPoseInferencer

# Initialize 3D pose inferencer
inferencer = MMPoseInferencer(
    pose3d='motionbert_dstformer-ft-243frm_8xb32-120e_h36m'
)

input_file_path = 'input_file.mp4'
output_dir_path = 'output_file'

# Run inference with automatic saving
result_generator = inferencer(
    inputs=input_file_path,
    show=True,
    out_dir=output_dir_path,
    return_vis=True,
    save_pred=True,
    disable_norm_pose_2d=False
)

while True:
    try:
        result = next(result_generator)
        print(result.keys())  # Do something with each result (e.g., process or save)
    except StopIteration:
        # If we reach the end of the generator, stop the loop
        print("No more results.")
        break  # should include 'keypoints', 'frame_id', etc.
