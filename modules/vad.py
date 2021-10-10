import torch
import torch.nn as nn
import torch.nn.functional as F


class down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, square_kernel=True):
        super().__init__()
        if square_kernel:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size, kernel_size), stride=(2, 2), dilation=1,
                          padding=(padding, padding), bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size, 1), stride=(2, 1), dilation=1, padding=(padding, 0), bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
            )

    def forward(self, x):
        x = self.down(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, square_kernel=True):
        super().__init__()
        if square_kernel:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(kernel_size, kernel_size), stride=(2, 2),
                                   padding=(padding, padding),
                                   output_padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(kernel_size, 1), stride=(2, 1), padding=(padding, 0),
                                   output_padding=(1, 0), bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x = torch.cat([x2, x1], dim=1)
        x = torch.add(x1, x2)
        return x


class out_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding):
        super().__init__()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(kernel_size, 1),
                               stride=(2, 1), padding=(padding, 0), output_padding=(1, 0)),
            # networks.BatchNorm2d(out_ch),
        )

    def forward(self, x1, x2):
        x1 = self.out_conv(x1)
        # x1 = torch.add(x1, x2)
        return x1


class UNetVad(nn.Module):
    def __init__(self, n_filters, nclass=1, norm_features=False):
        super().__init__()
        kernel_size = 5
        padding = 2
        self.norm_features = norm_features
        if norm_features:
            self.bn0 = nn.BatchNorm2d(23, affine=False)

        self.down1 = down(3, n_filters[0], kernel_size=kernel_size, padding=padding)
        self.down2 = down(n_filters[0], n_filters[1], kernel_size=kernel_size, padding=padding)
        self.down3 = down(n_filters[1], n_filters[2], kernel_size=kernel_size, padding=padding)
        self.down4 = down(n_filters[2], n_filters[3], kernel_size=kernel_size, padding=padding, square_kernel=False)
        self.down5 = down(n_filters[3], n_filters[4], kernel_size=kernel_size, padding=padding, square_kernel=False)

        self.up4 = up(n_filters[4], n_filters[-4], kernel_size=kernel_size, padding=padding, square_kernel=False)
        self.up3 = up(n_filters[-4], n_filters[-3], kernel_size=kernel_size, padding=padding, square_kernel=False)
        self.up2 = up(n_filters[-3], n_filters[-2], kernel_size=kernel_size, padding=padding)
        self.up1 = up(n_filters[-2], n_filters[-1], kernel_size=kernel_size, padding=padding)
        self.out_conv = out_conv(n_filters[-1], nclass, kernel_size=kernel_size, padding=padding)

        self.down_layers = ['down1', 'down2', 'down3', 'down4', 'down5']
        self.up_layers = ['up4', 'up3', 'up2', 'up1', 'out_conv']

    def _down(self, x):
        outputs = [x]
        for down_attr_name in self.down_layers:
            outputs.append(getattr(self, down_attr_name)(outputs[-1]))
        return outputs

    def _up(self, down_outputs):
        layer_output = down_outputs.pop()
        for _, up_attr_name in enumerate(self.up_layers):
            layer_output = getattr(self, up_attr_name)(layer_output, down_outputs[-1])
            down_outputs = down_outputs[:-1]
        return layer_output


    def forward(self, x):
        down_outuputs = self._down(x)
        x5 = self._up(down_outuputs)
        x = F.avg_pool2d(x5, kernel_size=(1, 12)).squeeze()
        return x

    def device(self):
        return next(self.parameters()).device

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
