import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import googlenet, GoogLeNet_Weights

class SimpleInceptionBackbone(nn.Module):
    def __init__(self, in_channels):
        super(SimpleInceptionBackbone, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),  # simulate large receptive field
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))  # global avg pooling
        )

    def forward(self, x):  # x: (B, C, H, W)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # (B, 256)
        return x

from torchvision.models import googlenet, GoogLeNet_Weights

class PretrainedInceptionBackbone(nn.Module):
    def __init__(self):
        super(PretrainedInceptionBackbone, self).__init__()
        weights = GoogLeNet_Weights.IMAGENET1K_V1
        model = googlenet(weights=weights, aux_logits=True)  # aux_logits=True as required
        
        self.feature_extractor = nn.Sequential(
            model.conv1,
            model.maxpool1,
            model.conv2,
            model.conv3,
            model.maxpool2,
            model.inception3a,
            model.inception3b,
            model.maxpool3,
            model.inception4a,
            model.inception4b,
            model.inception4c,
            model.inception4d,
            model.inception4e,
            model.maxpool4,
            model.inception5a,
            model.inception5b,
            model.avgpool  # Output: (B, 1024, 1, 1)
        )

        self.out_dim = 1024

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)  # (B, 1024)
        return x


class TwoStreamNetwork(nn.Module):
    def __init__(self, num_classes):
        super(TwoStreamNetwork, self).__init__()
        self.rgb_stream = PretrainedInceptionBackbone()
        self.flow_stream = SimpleInceptionBackbone(in_channels=20)

        fusion_dim = self.rgb_stream.out_dim + 256  # 1024 (RGB) + 256 (Flow)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb_input, flow_input):  # (B, T, C, H, W)
        B, T, C, H, W = rgb_input.shape

        rgb_input = rgb_input.view(B * T, C, H, W)
        flow_input = flow_input.view(B * T, 20, H, W)

        rgb_feat = self.rgb_stream(rgb_input)  # (B*T, 1024)
        flow_feat = self.flow_stream(flow_input)  # (B*T, 256)

        # Reshape back: (B, T, feat_dim)
        rgb_feat = rgb_feat.view(B, T, -1)
        flow_feat = flow_feat.view(B, T, -1)

        # Temporal average
        rgb_feat = rgb_feat.mean(dim=1)
        flow_feat = flow_feat.mean(dim=1)

        fused = torch.cat([rgb_feat, flow_feat], dim=1)  # (B, fusion_dim)

        out = self.classifier(fused)  # (B, num_classes)
        return out


# # Sample test
# model = TwoStreamNetwork(num_classes=10)
# rgb_dummy = torch.randn(2, 5, 3, 224, 224)
# flow_dummy = torch.randn(2, 5, 20, 224, 224)
# out = model(rgb_dummy, flow_dummy)
# print(out.shape)
