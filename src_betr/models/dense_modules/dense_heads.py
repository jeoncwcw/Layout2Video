from torch import nn

class DenseHeads(nn.Module):
    def __init__(self, heads, in_channels):
        super(DenseHeads, self).__init__()
        self.heads_dict = nn.ModuleDict()

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