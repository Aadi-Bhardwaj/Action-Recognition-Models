import os
import cv2
import numpy as np

def extract_optical_flow(video_path, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cap = cv2.VideoCapture(video_path)
    ret, prev = cap.read()
    if not ret:
        print(f"[ERROR] Could not read {video_path}")
        return

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)

        fx, fy = flow[..., 0], flow[..., 1]
        fx = np.clip((fx + 15) * 8.5, 0, 255).astype(np.uint8)
        fy = np.clip((fy + 15) * 8.5, 0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(out_dir, f"flow_x_{idx:05d}.jpg"), fx)
        cv2.imwrite(os.path.join(out_dir, f"flow_y_{idx:05d}.jpg"), fy)

        prev_gray = gray
        idx += 1

    cap.release()

# === Apply to All Videos Recursively ===

input_base = "D:\\TwoStream\\data"
output_base = os.path.join(input_base, "extracted_flow")  # create parallel flow_frames directory

splits = ['train', 'val', 'test']

for split in splits:
    split_dir = os.path.join(input_base, split)
    for class_name in os.listdir(split_dir):
        class_path = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for filename in os.listdir(class_path):
            if filename.endswith(('.avi', '.mp4')):
                video_path = os.path.join(class_path, filename)
                video_name = os.path.splitext(filename)[0]

                output_dir = os.path.join(output_base, split, class_name, video_name)
                


                print(f"➡️  Extracting flow for: {video_path} → {output_dir}")
                extract_optical_flow(video_path, output_dir)
