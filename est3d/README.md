# INSTRUCTION
1. MMPOSE 2D KEYPOINTS DETECTION:
From conda prompt with the "openmmlab" env activated:
python inf.py

2. GET MODIFIED MMPOSE JSON PREDICTIONS / GET INTERMEDIATE-FORMAT JSON FILES WITH TRACK IDS ADDED / PERFORM TRACKING AND GET JSONS WITH TRACK IDS:
From conda prompt with the "boxmot" env activated:
python bbox.py

3. TRANSLATION TO ALPHAPOSE FORMAT AND OBTAINING MULTIPLE [SHIFTED] SEQUENCES SIMULTANEOUSLY FROM MODIFIED MMPOSE JSON PREDICTIONS, GET MAIN PERSON TRACK IN ALPHAPOSE FORMAT:
From VSCode launched via "code" from conda prompt with OPENMMLAB env activated:
python best_to_alpha.py

4. CUT VIDEO TO FEED THE METADATA INTO MOTIONBERT:
From anaconda prompt with the activated "motionbert" env:
ffmpeg -i input.mp4 -ss 00:00:02.700 -c:v libx264 -c:a aac output.mp4

5. MOTIONBERT INFERENCE ON 2D KEYPOINTS:
From anaconda prompt with the activated "motionbert" env:
python infer_wild.py --vid_path input_video.mp4 --json_path input_json.json --out_path output_video_and_keypoints

6. GET NORMALIZED ENSEMBLED PREDICTIONS FROM MOTIONBERT:
From conda prompt with the "openmmlab" env activated:
python inspect_mbert.py

*---FURTHER PROCESSING IF 2 CAMERAS ARE AVAILABLE---*

7. GET TWO 3D TRAJECTORIES ALIGNED VIA PROCRUSTES
From conda prompt with the "openmmlab" env activated:
python apply_procrustes.py

8. FUSE THE TWO 3D TRAJECTORIES INTO ONE
From conda prompt with the "openmmlab" env activated:
python apply_procrustes.py
