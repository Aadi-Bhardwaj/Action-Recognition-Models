import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

class TwoStreamDataset(Dataset):
    def __init__(self, csv_file, rgb_root, flow_root, label_map, 
                 num_rgb_frames=5, rgb_stride=10, 
                 num_flow_stack=5, flow_stack_depth=10, 
                 rgb_transform=None, flow_transform=None):
        self.df = pd.read_csv(csv_file)
        self.rgb_root = rgb_root
        self.flow_root = flow_root
        self.label_map = label_map

        self.num_rgb = num_rgb_frames
        self.rgb_stride = rgb_stride

        self.num_flow = num_flow_stack
        self.flow_stack_depth = flow_stack_depth

        self.rgb_transform = rgb_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

        self.flow_transform = flow_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip_path = row['clip_path']
        label = self.label_map[row['label']]
        video_name = row['clip_name']

        parts = clip_path.strip('/').split('/')
        split, class_name = parts[0], parts[1]

        ### RGB frames
        rgb_dir = os.path.join(self.rgb_root, split, class_name, video_name)
        rgb_frames = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
        rgb_imgs = []

        for i in range(self.num_rgb):
            frame_idx = i * self.rgb_stride
            if frame_idx >= len(rgb_frames): 
                frame_idx = -1
            frame_path = os.path.join(rgb_dir, rgb_frames[frame_idx])
            img = Image.open(frame_path).convert("RGB")
            rgb_imgs.append(self.rgb_transform(img))

        rgb_tensor = torch.stack(rgb_imgs)  # shape: (num_rgb, 3, H, W)

        ### Optical Flow
        flow_dir = os.path.join(self.flow_root, split, class_name, video_name)
        flow_x_files = sorted([f for f in os.listdir(flow_dir) if 'flow_x' in f])
        flow_y_files = sorted([f for f in os.listdir(flow_dir) if 'flow_y' in f])
        flow_stacks = []

        for i in range(self.num_flow):
            start = i * self.flow_stack_depth
            stack = []
            for j in range(self.flow_stack_depth):
                flow_idx = start + j

                if flow_idx >= len(flow_x_files):
                    # Pad with zeros if out of bounds
                    fx = torch.zeros((1, 224, 224))
                    fy = torch.zeros((1, 224, 224))
                else:
                    fx = Image.open(os.path.join(flow_dir, flow_x_files[flow_idx])).convert('L')
                    fy = Image.open(os.path.join(flow_dir, flow_y_files[flow_idx])).convert('L')
                    fx = self.flow_transform(fx)
                    fy = self.flow_transform(fy)
                
                stack.extend([fx, fy])  # 2 channels per time step

            flow_stack = torch.cat(stack, dim=0)  # shape: (2 * depth, H, W)
            flow_stacks.append(flow_stack)

        flow_tensor = torch.stack(flow_stacks)  # shape: (num_flow, 2*depth, H, W)

        # rgb_tensor: (5, 3, 224, 224), flow_tensor: (5, 20, 224, 224), label: int
        return rgb_tensor, flow_tensor, label
