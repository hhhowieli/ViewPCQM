import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class SetPooling(nn.Module):
    def __init__(self, plane) -> None:
        super().__init__()

        self.pooling_convert = nn.Sequential(
            nn.Conv2d(in_channels=plane*3, out_channels=plane, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=plane, out_channels=plane, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, view_set, p):

        sum_x = torch.sum(view_set, dim=1)
        max_x = torch.max(view_set, dim=1)[0]
        lp_x =  torch.norm(view_set, p=p, dim=1)

        aggr_x = torch.concat([sum_x, lp_x, max_x], dim=1)

        pooling_x = self.pooling_convert(aggr_x)

        return pooling_x



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

class CrossViewSpatialFusion(nn.Module):
    def __init__(self, plane, hidp, patch_size, rates, num_views) -> None:
        super().__init__()

        self.num_views = num_views
        self.hidp = hidp
        self.ps = patch_size

        self.activation = nn.Sigmoid()
        # self.activation = nn.Softmax(-1)

        self.patch_aggr = MultiFieldConv2d(inp=plane, outp=3*hidp, k=patch_size, rates=rates, hidp=plane*3)

        self.to_kv = nn.Sequential(
            nn.Conv2d(in_channels=hidp, out_channels=hidp, kernel_size=patch_size, padding=patch_size//2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=patch_size, stride=patch_size),
        )

        self.to_out = nn.Sequential(
            nn.Conv2d(in_channels=hidp, out_channels=plane, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=plane, out_channels=plane, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.scale = hidp ** (-0.5)
        self.scales = nn.Parameter(torch.ones(num_views), requires_grad=True)


    def forward(self, x):
        B, N, C, H, W = x.shape

        x = rearrange(x, "b n c h w -> (b n) c h w")

        # x_c = self.aggr(x)

        x_qkv = F.relu(self.patch_aggr(x), inplace=True)
        x_q, x_k, x_v = x_qkv.chunk(3, dim=1)

        x_k_, x_v_ = self.to_kv(x_k), self.to_kv(x_v)

        x_v_ = x_v_ * (1 / self.ps**2)

        x_v = rearrange(x_v, "(b n) c h w -> b n c h w", n=self.num_views)
        # print(x_v.shape)

        Qs = rearrange(x_q, "(b n) c h w -> b n (h w) c", b=B, n=self.num_views)
        Ks = rearrange(x_k_, "(b n) c h w -> b n (h w) c", b=B, n=self.num_views)
        Vs = rearrange(x_v_, "(b n) c h w -> b n (h w) c", b=B, n=self.num_views)

        res = []
        attns = []
        for i in range(self.num_views):
            q = Qs[:, i, :, :]
            k = torch.concat([Ks[:, :i, :, :], Ks[:, i+1:, :, :]], dim=1)
            k = rearrange(k, "b n nv d -> b (n nv) d", n=self.num_views-1, d=self.hidp)

            v = torch.concat([Vs[:, :i, :, :], Vs[:, i+1:, :, :]], dim=1)
            v = rearrange(v, "b n nv d -> b (n nv) d", n=self.num_views-1, d=self.hidp)

            # print(q.shape)
            # print(k.shape)

            inter = q @ k.transpose(-1, -2) * self.scale
            # print(inter)

            attn = self.activation(inter)
            # print(attn)
            attns.append(rearrange(attn, "b g (n nv) -> b n g nv", n=self.num_views-1))

            res.append((attn @ v))

        res = torch.stack(res, dim=1)

        res = rearrange(res, "b n (h w) c -> b n c h w", n=self.num_views, h=H)

        res = F.relu(res + self.scales[None, :, None, None, None] * x_v, inplace=True)

        res = self.to_out(rearrange(res, "b n c h w -> (b n) c h w", n=self.num_views, h=H)) 

        res = rearrange(res, "(b n) c h w -> b n c h w", n=N)

        return res




class ViewPooling(nn.Module):
    def __init__(self, plane, hidp, patch_size, rates, num_views) -> None:
        super().__init__()

        self.num_views = num_views

        self.cvsa = CrossViewSpatialFusion(
            plane=plane,
            hidp=hidp,
            patch_size=patch_size,
            rates=rates,
            num_views=3
        )

        self.set_pooling = SetPooling(plane=plane)

        # self.p = nn.Parameter(torch.tensor([1., 1.]), requires_grad=True)

    def forward(self, x):

        # self.p = torch.clip(self.p, 0, 1)

        x = rearrange(x, "(b n) c h w -> b n c h w", n=self.num_views)

        top_bottom_x = x[:, :2, :, :, :]
        left_right_x = x[:, 2:4, :, :, :]
        front_back_x = x[:, 4:, :, :, :]

        x_bt = self.set_pooling(top_bottom_x, p=2*(2**0.5))
        x_rl = self.set_pooling(left_right_x, p=2*(2**0.5))
        x_bf = self.set_pooling(front_back_x, p=2*(2**0.5))
        # x_all = self.set_pooling1(x, p=4)

        new_x = torch.stack([x_bt, x_rl, x_bf], dim=1)
        # cvsf, attns = self.cvsa(new_x)
        cvsf = self.cvsa(new_x)
        new_x = new_x + cvsf

        pooled_x = self.set_pooling(new_x, p=2*(2**0.5))

        # return pooled_x, attns
        return pooled_x
