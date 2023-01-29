import torch
from torch import nn

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()

    channels_per_group = num_channels // groups

    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    x = x.view(batchsize, -1, height, width)

    return x

class shufflenet_unit(nn.Module):
    def __init__(self, groups, in_channels, mid_channels, out_channels, stride, first_pw=False):
        super(shufflenet_unit, self).__init__()

        self.groups = groups
        self.stride = stride

        if stride == 2:
            out_channels = out_channels - in_channels

        self.GConV_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=mid_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      groups=1 if first_pw else groups,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.DWConv = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels,
                      out_channels=mid_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      groups=mid_channels,
                      bias=False),
            nn.BatchNorm2d(mid_channels)
        )

        self.GConV_2 = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=3,
                                     stride=2,
                                     padding=1)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        short_cut = x

        x = self.GConV_1(x)
        x = channel_shuffle(x, self.groups)
        x = self.DWConv(x)
        x = self.GConV_2(x)

        if self.stride == 1:
            out = x + short_cut

        if self.stride == 2:
            short_cut = self.avg_pool(short_cut)
            out = torch.cat([x, short_cut], dim=1)

        out = self.act(out)
        return out

def chose_channels(groups):
    stages = []
    if groups == 3:
        stages = [
            [4, 24, 240],
            [8, 240, 480],
            [4, 480, 960]
        ]

    if groups == 8:
        stages = [
            [4, 24, 384],
            [8, 384, 768],
            [4, 768, 1536]
        ]
    if groups not in [3, 8]:
        raise IndexError
    return stages

class shufflenet(nn.Module):
    def __init__(self, groups, ratio=1.0, num_classes=1000, dropout=0.2):
        super(shufflenet, self).__init__()

        self.ratio = ratio

        if ratio == 1.5 and groups == 8:
            self.ratio = 1.0

        if ratio == 0.5 and groups == 8:
            self.ratio = 0.67


        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=int(self.ratio * 24),
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(int(self.ratio * 24)),
            nn.ReLU(inplace=True)
        )

        self.max_pool = nn.MaxPool2d(kernel_size=3,
                                     padding=1,
                                     stride=2)

        self.stages = chose_channels(groups)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(int(self.stages[2][2] * ratio), num_classes)

        layers = []
        block = shufflenet_unit

        for stage in range(len(self.stages)):
            num_blocks, in_channels, out_channels = self.stages[stage]
            in_channels = int(in_channels * ratio)
            out_channels = int(out_channels * ratio)

            for num_layer in range(num_blocks):

                if num_layer == 0:
                    if stage == 0:
                        if ratio == 1.5 and groups == 8:
                            in_channels = 24
                        if ratio == 0.5 and groups == 8:
                            in_channels = 16
                        layers.append(block(groups=groups,
                                            in_channels=in_channels,
                                            out_channels=out_channels,
                                            mid_channels=out_channels // 4,
                                            stride=2,
                                            first_pw=True))
                    else:
                        layers.append(block(groups=groups,
                                            in_channels=in_channels,
                                            out_channels=out_channels,
                                            mid_channels=out_channels // 4,
                                            stride=2))
                else:
                    layers.append(block(groups=groups,
                                        in_channels=in_channels,
                                        out_channels=out_channels,
                                        mid_channels=out_channels // 4,
                                        stride=1))
                in_channels = out_channels

        self.shuffle_units = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.max_pool(x)

        x = self.shuffle_units(x)

        x = self.global_pool(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        x = self.classifier(x)
        return x

def ShuffleNet_050_g3(**kwargs):
    return shufflenet(groups=3, ratio=0.5, **kwargs)

def ShuffleNet_050_g8(**kwargs):
    return shufflenet(groups=8, ratio=0.5, **kwargs)

def ShuffleNet_100_g3(**kwargs):
    return shufflenet(groups=3, **kwargs)

def ShuffleNet_100_g8(**kwargs):
    return shufflenet(groups=8, **kwargs)

def ShuffleNet_150_g3(**kwargs):
    return shufflenet(groups=3, ratio=1.5, **kwargs)

def ShuffleNet_150_g8(**kwargs):
    return shufflenet(groups=8, ratio=1.5, **kwargs)

def ShuffleNet_200_g3(**kwargs):
    return shufflenet(groups=3, ratio=2.0, **kwargs)

def ShuffleNet_200_g8(**kwargs):
    return shufflenet(groups=8, ratio=2.0, **kwargs)

if __name__ == "__main__":
    x = torch.randn(1, 3, 128, 128)
    net = ShuffleNet_200_g8(num_classes=7)
    net.eval()
    print(net(x).shape)





