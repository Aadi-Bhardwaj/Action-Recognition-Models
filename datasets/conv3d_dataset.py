import os
import csv
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import random

class UCF101Dataset(Dataset):
    def __init__(self, root_dir, split_file, num_frames=16, transform=None, num_classes=10):
        self.samples = []
        self.num_frames = num_frames
        self.transform = transform
        self.root_dir = root_dir

        # Read CSV and collect class names
        with open(split_file, 'r') as f:
            reader = csv.DictReader(f)
            class_names = []
            raw_samples = []

            for row in reader:
                clip_path = row['clip_path'].strip()
                label_name = row['label'].strip()

                # Limit to first `num_classes`
                if label_name not in class_names:
                    if len(class_names) >= num_classes:
                        continue
                    class_names.append(label_name)

                raw_samples.append((clip_path, label_name))

        # Class name → index mapping
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
        self.selected_classes = set(class_names)

        # Build final (video_path, label_index) list
        for clip_path, label_name in raw_samples:
            if label_name not in self.selected_classes:
                continue
            label = self.class_to_idx[label_name]
            video_path = os.path.join(self.root_dir, clip_path.lstrip('/'))
            self.samples.append((video_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._load_frames(video_path)  # shape: [T, C, H, W]
        frames = frames.permute(1, 0, 2, 3)      # [C, T, H, W]
        return frames, label


    def _load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = transforms.ToTensor()(frame)
            frames.append(frame)

        cap.release()

        # Clip or pad to num_frames
        if len(frames) >= self.num_frames:
            start = random.randint(0, len(frames) - self.num_frames)
            frames = frames[start:start + self.num_frames]
        else:
            while len(frames) < self.num_frames:
                frames.append(frames[-1])

        return torch.stack(frames)  # [T, C, H, W]
