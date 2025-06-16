
# Initialize 3D pose inferencer with MotionBERT (pretrained on Human3.6M)

from mmpose.apis import MMPoseInferencer

# Initialize 3D pose inferencer
inferencer = MMPoseInferencer(
    #pose2d='rtmpose-l_8xb512-700e_body8-halpe26-384x288'
    #pose2d='rtmpose-m_8xb512-700e_body8-halpe26-256x192',  # explicitly set Halpe26 format
    pose3d='motionbert_dstformer-ft-243frm_8xb32-120e_h36m'
    #pose2d='rtmpose-l_8xb256-420e_coco-256x192'
)

# Run inference with automatic saving
result_generator = inferencer(
    inputs='neurologist/file_469.mp4',
    show=True,
    out_dir='output_test_mbert_469',
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
