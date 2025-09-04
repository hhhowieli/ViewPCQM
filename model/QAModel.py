import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import BasicBlock, conv1x1, Bottleneck
from .view_base import ViewPooling

from einops import rearrange

def editWeight(ckpt):
    new_ckpt = {}

    for k in ckpt:
        if "fc" not in k:
            new_ckpt[k] = ckpt[k]

            if "layer" in k:
                new_ckpt["vp_" + k] = ckpt[k]

    return new_ckpt


def init_weights(net):

    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class QAModel(nn.Module):
    def __init__(self,
            block,
            layers,
            patch_sizes,
            norm_layer=nn.BatchNorm2d,
            num_views=6,
            groups=1,

        ) -> None:
        super().__init__()

        self.num_views = num_views

        self._norm_layer = norm_layer

        # Resnet
        self.inplanes = 64
        self.groups = 1
        self.dilation = 1
        self.base_width = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, 64, layers[0])

        self.vp1 = ViewPooling(64, 4*64, patch_sizes[0], rates=3, num_views=self.num_views)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.vp2 = ViewPooling(128, 4*128, patch_sizes[1], rates=2, num_views=self.num_views)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.vp3 = ViewPooling(256, 2*256, patch_sizes[2], rates=2, num_views=self.num_views)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.vp4 = ViewPooling(512, 2*512, patch_sizes[3], rates=1, num_views=self.num_views)

        self.inplanes = 64
        self.vp_layer1 = self._make_layer(block, 64, layers[0])
        self.vp_layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.vp_layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.vp_layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.regressor = nn.Sequential(
            nn.Linear(in_features=2*(512+256+128+64)*block.expansion, out_features=2*720),
            nn.LayerNorm(2*720),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=2*720, out_features=2*72),
            nn.LayerNorm(2*72),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2*72, 1)
        )

        init_weights(self)


    def forward(self, ims, return_attn=False):

        ft = []
        ft_mv = []

        ims = rearrange(ims, "b n c h w -> (b n) c h w", n=self.num_views)

        x = self.conv1(ims)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # px, _ = self.vp1(x)
        px = self.vp1(x)
        ft.append(F.adaptive_avg_pool2d(px, 1).flatten(start_dim=1))
        ft_mv.append(F.adaptive_avg_pool2d(x, 1).flatten(start_dim=1))

        x = self.layer2(x)
        # vpx, attns = self.vp2(x)
        vpx = self.vp2(x)
        px = vpx + self.vp_layer2(px)
        ft.append(F.adaptive_avg_pool2d(px, 1).flatten(start_dim=1))
        ft_mv.append(F.adaptive_avg_pool2d(x, 1).flatten(start_dim=1))

        x = self.layer3(x)
        px = self.vp3(x) + self.vp_layer3(px)
        ft.append(F.adaptive_avg_pool2d(px, 1).flatten(start_dim=1))
        ft_mv.append(F.adaptive_avg_pool2d(x, 1).flatten(start_dim=1))

        x = self.layer4(x)
        # vpx, _ = self.vp4(x)
        vpx = self.vp4(x)
        px = vpx + self.vp_layer4(px)
        ft.append(F.adaptive_avg_pool2d(px, 1).flatten(start_dim=1))
        ft_mv.append(F.adaptive_avg_pool2d(x, 1).flatten(start_dim=1))

        ft = torch.concat(ft, dim=-1)

        ft_mv = torch.concat(ft_mv, dim=-1)
        ft_mv = rearrange(ft_mv, "(b n) d -> b n d", n=self.num_views)
        ft_mv = torch.max(ft_mv, dim=1)[0]

        ft = torch.concat([ft, ft_mv], dim=-1)

        qual = self.regressor(ft)

        # if return_attn:
        #     return qual, attns

        return qual

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,

    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)



def qamodel_resnet34(
        pretrained=True,
        norm_layer=nn.BatchNorm2d,
    ):
    model = QAModel(
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        patch_sizes=[7, 5, 3, 1],
        norm_layer=norm_layer
    )

    if pretrained:
        ckpt = torch.load("./checkpoints/resnet34-b627a593.pth")
        new_ckpt = editWeight(ckpt)
        model.load_state_dict(new_ckpt, strict=False)

    return model



