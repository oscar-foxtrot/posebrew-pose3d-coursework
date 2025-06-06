# INSTRUCTION
1. MMPOSE 2D KEYPOINTS DETECTION:<br>
From conda prompt with the "openmmlab" env activated:<br>
python inf.py<br>

2. GET MODIFIED MMPOSE JSON PREDICTIONS / GET INTERMEDIATE-FORMAT JSON FILES WITH TRACK IDS ADDED / PERFORM TRACKING AND GET JSONS WITH TRACK IDS:<br>
From conda prompt with the "boxmot" env activated:<br>
python bbox.py<br>

3. TRANSLATION TO ALPHAPOSE FORMAT AND OBTAINING MULTIPLE [SHIFTED] SEQUENCES SIMULTANEOUSLY FROM MODIFIED MMPOSE JSON PREDICTIONS, GET MAIN PERSON TRACK IN ALPHAPOSE FORMAT:<br>
From conda prompt with the "openmmlab" env activated:<br>
python intermediate_to_alpha.py<br>

4. CUT VIDEO TO FEED THE METADATA INTO MOTIONBERT (CUSTOMIZE THE TIME):<br>
From anaconda prompt with the activated "motionbert" env:<br>
ffmpeg -i input.mp4 -ss 00:00:02.700 -c:v libx264 -c:a aac output.mp4<br>

5. MOTIONBERT INFERENCE ON 2D KEYPOINTS:<br>
From anaconda prompt with the activated "motionbert" env, from inside the MotionBERT repository:<br>
python infer_wild.py --vid_path input_video.mp4 --json_path input_json.json --out_path output_video_and_keypoints<br>

6. GET NORMALIZED ENSEMBLED PREDICTIONS FROM MOTIONBERT:<br>
From conda prompt with the "openmmlab" env activated:<br>
python inspect_mbert.py<br>


     
*---FURTHER PROCESSING IF 2 CAMERAS ARE AVAILABLE---*<br>

7. GET TWO 3D TRAJECTORIES ALIGNED VIA PROCRUSTES<br>
From conda prompt with the "openmmlab" env activated:<br>
python apply_procrustes.py<br>

8. FUSE THE TWO 3D TRAJECTORIES INTO ONE<br>
From conda prompt with the "openmmlab" env activated:<br>
python combine_3d.py<br>
