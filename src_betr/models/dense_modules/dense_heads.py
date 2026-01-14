from torch import nn
import torch

class DenseHeads(nn.Module):
    def __init__(self, heads, in_channels):
        super(DenseHeads, self).__init__()
        self.heads_dict = nn.ModuleDict()
        
        # TODO: Fix hard-coded mean and std values
        bb8_offset_mean = torch.tensor([
            -0.0015, -0.0712, -0.0170, -0.0658, -0.0155,  0.0640, -0.0007,  0.0571,
             0.0142, -0.0634,  0.0053, -0.0576,  0.0031,  0.0721,  0.0121,  0.0649
        ])
        
        bb8_depth_mean = torch.tensor([
            -0.0164, -0.0241,  0.0098,  0.0147, -0.0227, -0.0306,  0.0035,  0.0084
        ])

        for role in heads:
            if role in ['center heatmap', "center depth"]:
                out_ch = 1
            elif role == "bb8 offsets":
                out_ch = 16  # 8 corners * 2 (x, y)
            elif role == "bb8 depth offsets":
                out_ch = 8  # 8 corners
            else:
                raise ValueError(f"Unknown head role: {role}")
            
            fc = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_ch, kernel_size=1)
            )
            self._init_weights(fc)
            
            if "heatmap" in role:
                fc[-1].bias.data.fill_(-2.19)  # Initialize heatmap head bias
            else:
                self._init_weights(fc)
                

            self.heads_dict[role] = fc

    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        outputs = {}
        for role, head in self.heads_dict.items():
            outputs[role] = head(x)
        return outputs