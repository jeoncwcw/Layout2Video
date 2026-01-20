from torch import nn
import torch

class DenseHeads(nn.Module):
    def __init__(self, heads, in_channels):
        super(DenseHeads, self).__init__()
        self.heads_dict = nn.ModuleDict()

        for role in heads:
            if role in ['corner heatmaps']:
                out_ch = 8  # 8 corners
            elif role in ['corner depths']:
                out_ch = 8  # Depth for each corner
            else:
                raise ValueError(f"Unknown head role: {role}")
            
            fc = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_ch, kernel_size=1)
            )
            
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