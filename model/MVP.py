import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class DilatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate):
        super(DilatedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2) * dilation_rate, dilation=dilation_rate)

    def forward(self, x):
        return self.conv(x)


class MultiFieldConv2d(nn.Module):
    def __init__(self, inp, outp, k, rates, hidp=0) -> None:
        super().__init__()

        self.conv_layers = nn.ModuleList()

        self.rates = rates

        if hidp == 0:
            hidp = inp

        for i in range(1, rates+1):
            self.conv_layers.append(
                nn.Sequential(
                    DilatedConv2d(in_channels=inp, out_channels=hidp, kernel_size=k, dilation_rate=i),
                )
            )

        self.convert = nn.Sequential(
            nn.Conv2d(in_channels=hidp*rates, out_channels=outp, kernel_size=1),
            nn.BatchNorm2d(outp),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outp, out_channels=outp, kernel_size=1),
        )

    def forward(self, x):
        U = []

        for layer in self.conv_layers:
            U.append(layer(x))

        U = torch.concat(U, dim=1)
        V = self.convert(U)

        return V


class SetPooling(nn.Module):
    def __init__(self, plane) -> None:
        super().__init__()

        self.pooling_convert = nn.Sequential(
            nn.Conv2d(in_channels=plane*3, out_channels=plane, kernel_size=1),
            # nn.BatchNorm2d(plane),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=plane, out_channels=plane, kernel_size=1),
            # nn.BatchNorm2d(plane),
            nn.ReLU(inplace=True),
        )

    def forward(self, view_set, p):

        sum_x = torch.sum(view_set, dim=1)
        max_x = torch.max(view_set, dim=1)[0]
        lp_x =  torch.norm(view_set, p=p, dim=1)

        aggr_x = torch.concat([sum_x, lp_x, max_x], dim=1)

        pooling_x = self.pooling_convert(aggr_x)

        return pooling_x

class ViewPooling(nn.Module):
    def __init__(self, plane, hidp, patch_size, rates, num_views) -> None:
        super().__init__()

        self.set_pooling = SetPooling(hidp)

        self.aggr = MultiFieldConv2d(inp=plane, outp=hidp, k=patch_size, rates=rates)

        # self.to_q = nn.Sequential(
        #     nn.Conv2d(in_channels=hidp, out_channels=hidp, kernel_size=1),
        #     nn.ReLU(inplace=True),
        # )

        self.to_kv = nn.Sequential(
            nn.Conv2d(in_channels=hidp, out_channels=2*hidp, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=patch_size, stride=patch_size),
        )

        self.scale = hidp ** (-0.5)

        self.num_views = num_views

        self.ps = patch_size

        self.activate = nn.Softmax(-1)

        self.to_out = nn.Sequential(
            nn.Conv2d(in_channels=hidp, out_channels=hidp, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidp, out_channels=plane, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, views):
        """
            views: (b n) c h w
        """
        _, C, H, W = views.shape


        aggr_views = F.relu(
            self.aggr(views),
            inplace=True
        )

        pooled_view = self.set_pooling(rearrange(aggr_views, "(b n) c h w -> b n c h w", n=self.num_views), p=2)
        kv = self.to_kv(aggr_views)
        x_k, x_v = kv.chunk(2, dim=1)
        x_v = x_v * (1 / self.ps**2)

        Qs = rearrange(pooled_view, "b c h w -> b (h w) c")
        Ks = rearrange(x_k, "(b n) c h w -> b (n h w) c", n=self.num_views)
        Vs = rearrange(x_v, "(b n) c h w -> b (n h w) c", n=self.num_views)

        attn = self.activate(Qs @ Ks.transpose(-1, -2) * self.scale)

        res = attn @ Vs

        pooled_view = F.relu(pooled_view + rearrange(res, "b (h w) c -> b c h w", h=H), inplace=True)

        res = self.to_out(pooled_view)

        return res


        





