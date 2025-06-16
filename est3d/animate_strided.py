import cv2
import os

image_folder = 'StridedTransformer-Pose3D/demo/output_1/file_469/pose2D'
output_video = 'STRIDED_ANIM.mp4'     # Output video filename
fps = 30                             # Frames per second of the output video

# Get sorted list of image files
images = sorted([img for img in os.listdir(image_folder) if img.endswith('_2D.png')])

if not images:
    raise ValueError("No images found in folder!")

# Read first image to get the size
first_frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also try 'XVID' or 'avc1'
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for image_file in images:
    img_path = os.path.join(image_folder, image_file)
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Warning: Could not read {img_path}, skipping.")
        continue
    video.write(frame)

video.release()
print(f"Video saved as {output_video}")