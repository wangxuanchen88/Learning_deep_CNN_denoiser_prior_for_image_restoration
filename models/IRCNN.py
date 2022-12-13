import torch.nn as nn


class IRCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64):
        """
        # ------------------------------------
        denoiser of IRCNN
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        """
        super(IRCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=(3, 3), stride=(1, 1), padding=1, dilation=(1, 1),
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=(3, 3), stride=(1, 1), padding=2, dilation=(2, 2),
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=(3, 3), stride=(1, 1), padding=3, dilation=(3, 3),
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=(3, 3), stride=(1, 1), padding=4, dilation=(4, 4),
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=(3, 3), stride=(1, 1), padding=3, dilation=(3, 3),
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=(3, 3), stride=(1, 1), padding=2, dilation=(2, 2),
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=(3, 3), stride=(1, 1), padding=1,
                      dilation=(1, 1), bias=True),
        )

    def forward(self, x):
        noise = self.model(x)
        denoised_image = x - noise
        return noise, denoised_image
