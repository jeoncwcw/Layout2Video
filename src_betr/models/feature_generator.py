import torch
import torch.nn as nn

class FeatureGenerator(nn.Module):
    def __init__(self):
        super(FeatureGenerator, self).__init__()
        # Initialize layers or parameters here
        self.patchify_conv_mono = nn.Conv2d(
            in_channels=1,
            out_channels=512,
            kernel_size=14,
            stride=14,
        )
        self.patchify_conv_metric = nn.Conv2d(
            in_channels=1,
            out_channels=512,
            kernel_size=14,
            stride=14,
        )
        self.norm_metric = nn.InstanceNorm2d(1, affine=True)
        self.norm_mono = nn.InstanceNorm2d(1, affine=True)
    def forward(self, o_metric, o_mono, o_dino3):
        patchified_metric = self.patchify_conv_metric(self.norm_metric(torch.log1p(o_metric)))
        patchified_mono = self.patchify_conv_mono(self.norm_mono(o_mono))
        da3_features = torch.concat([patchified_metric, patchified_mono], dim=1)
        combined_features = torch.concat([da3_features, o_dino3], dim=1)
        return combined_features