import os
import subprocess

input_base = "D:\\TwoStream\\data"         # path to your 'data' folder
output_base = "D:\\TwoStream\\data\\extracted"  # where you want to save RGB frames
frame_rate = 25                         # can adjust

def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    command = [
        'ffmpeg',
        '-i', video_path,
        '-q:v', '2',
        '-vf', f'fps={frame_rate}',
        os.path.join(output_folder, 'frame_%05d.jpg')
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Go through train, val, test
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(input_base, split)
    for class_name in os.listdir(split_dir):
        class_path = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for video_file in os.listdir(class_path):
            if video_file.endswith(('.avi', '.mp4')):
                video_path = os.path.join(class_path, video_file)
                video_name = os.path.splitext(video_file)[0]

                # Build output path: rgb_frames/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/
                output_folder = os.path.join(output_base, split, class_name, video_name)

                # print(video_path)
                # print(output_folder)

                print(f"Extracting: {video_path} → {output_folder}")
                extract_frames(video_path, output_folder)
