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

            # if k[:5] == "conv1":
                # new_ckpt[k] = ckpt[k].repeat(1, 2, 1, 1)

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

        # self.conv1 = nn.Conv2d(6, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.regressor = nn.Sequential(
            nn.Linear(in_features=(512 + 256 + 128 + 64)*block.expansion, out_features=720*block.expansion),
            nn.LayerNorm(720*block.expansion),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=720*block.expansion, out_features=72*block.expansion),
            # nn.LayerNorm(72),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(72*block.expansion, 1)
        )

        init_weights(self)


    def forward(self, ims):

        ft = []


        ims = rearrange(ims, "b n c h w -> (b n) c h w", n=self.num_views)

        x = self.conv1(ims)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        x = self.layer1(x)
        ft.append(F.adaptive_avg_pool2d(x, 1).flatten(start_dim=1))

        x = self.layer2(x)
        ft.append(F.adaptive_avg_pool2d(x, 1).flatten(start_dim=1))

        x = self.layer3(x)
        ft.append(F.adaptive_avg_pool2d(x, 1).flatten(start_dim=1))

        x = self.layer4(x)
        ft.append(F.adaptive_avg_pool2d(x, 1).flatten(start_dim=1))
        # print(x.shape)

        ft = torch.concat(ft, dim=-1)

        ft = rearrange(ft, "(b n) d -> b n d", n=self.num_views)
        ft = torch.max(ft, dim=1)[0]

        qual = self.regressor(ft)

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

def qamodel_resnet18(
        pretrained=True,
        norm_layer=nn.BatchNorm2d,
    ):
    model = QAModel(
        block=BasicBlock, 
        layers=[2, 2, 2, 2], 
        norm_layer=norm_layer
    )

    if pretrained:
        ckpt = torch.load("./checkpoints/resnet18-5c106cde.pth")
        new_ckpt = editWeight(ckpt)
        model.load_state_dict(new_ckpt, strict=False)

    return model


def baseline_resnet34(
        pretrained=True,
        norm_layer=nn.BatchNorm2d,
    ):
    model = QAModel(
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        norm_layer=norm_layer
    )

    if pretrained:
        ckpt = torch.load("./checkpoints/resnet34-b627a593.pth")
        new_ckpt = editWeight(ckpt)
        model.load_state_dict(new_ckpt, strict=False)

    return model


def baseline_resnet50(
        pretrained=True,
        norm_layer=nn.BatchNorm2d,
    ):
    model = QAModel(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        norm_layer=norm_layer,
    )

    if pretrained:
        ckpt = torch.load("./checkpoints/resnet50-0676ba61.pth")
        new_ckpt = editWeight(ckpt)
        model.load_state_dict(new_ckpt, strict=False)

    return model

def qamodel_resnetX(
        pretrained=True,
        norm_layer=nn.BatchNorm2d,
        edit_norm=False
    ):
    model = QAModel(
        block=BasicBlock,
        layers=[5, 4, 6, 3],
        norm_layer=norm_layer
    )

    return model

# def qamodel_resnet(
#         view_arch,
#         vp_arch,
#         view_layers,
#         vp_layers
#     ):
#     basicblock = ["resnet18", "resnet34"]
#     # bottleblock = ["resnet50", "resnet101"]
    
#     view_block = BasicBlock if view_arch in basicblock else Bottleneck
#     vp_block = BasicBlock if vp_arch in basicblock else Bottleneck
#     model = QAModel(
#         block=view_block,
#         layers=view_layers,
#         vp_block=vp_block,
#         vp_layers=vp_layers,
#     )
#     if view_arch == "resnet18":
#         view_ckpt = torch.load("./checkpoints/resnet18-5c106cde.pth")
#     elif view_arch == "resnet34":
#         view_ckpt = torch.load("./checkpoints/resnet34-b627a593.pth")
#     elif view_arch == "resnet50":
#         view_ckpt = torch.load("./checkpoints/resnet50-19c8e357.pth")
    
#     if vp_arch == "resnet18":
#         vp_ckpt = torch.load("./checkpoints/resnet18-5c106cde.pth")
#     elif vp_arch == "resnet34":
#         vp_ckpt = torch.load("./checkpoints/resnet34-b627a593.pth")
#     elif vp_arch == "resnet50":
#         vp_ckpt = torch.load("./checkpoints/resnet50-19c8e357.pth")

#     ckpt = editWeight(view_ckpt, vp_ckpt)

#     model.load_state_dict(ckpt, strict=False)

#     return model


if __name__=="__main__":
    # ckpt = torch.load("./checkpoints/resnet34-b627a593.pth")
    
    # editWeight(ckpt)

    from torchvision import models

    resnet50 = models.resnet50()

    print(resnet50)



