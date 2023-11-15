import torch
import torch.nn as nn
import numpy as np
from basisNet import *


def make_bilinear_weights(size, num_channels):
    ''' Make a 2D bilinear kernel suitable for upsampling
    Stack the bilinear kernel for application to tensor '''
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

    # print filt
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, 1, size, size)
    for i in range(num_channels):
        w[i, 0] = filt
    return w

class DecoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels, upsample_type ='deconv'):
        super(DecoderBlock, self).__init__()
        """
        4.4.4 speed analysis
        For fair comparison, we use bilinear interpolation based
        up-sampling instead of de-convolution in PortraitNet. We find
        that PortraitNet achieves a good balance between accuracy and
        efficiency.
        """
        assert upsample_type in ['deconv', 'bilinear']
        if upsample_type == 'deconv':
            self.upsample = nn.ConvTranspose2d(in_channels=output_channels,out_channels=output_channels,kernel_size=4,stride=2,padding=1,bias=False)
        else:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.d_block = d_block(in_channel=input_channels,out_channel=output_channels)

    def forward(self, x):
        out = self.d_block(x)
        out = self.upsample(out)
        return out


class ProtraitNet(nn.Module):
    def __init__(self, n_class = 2):
        '''
        :param n_class:  class number of the segmentation
        :param input_size: the image size
        '''
        super(ProtraitNet, self).__init__()

        # -----encoder---------
        # self.conv1 = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=1,stride=1)
        block = inverted_res_block
        inverted_residual_setting= [
            # t, c, n, s of MobileNetV2
            [1, 16, 1, 1],  # e-stage1 1/2 channel = 16
            [6, 24, 2 ,2],  # e-stage2 1/4 channel = 24
            [6, 32, 3, 2],  # e-stage3 1/8 channel = 32
            [6, 64, 4, 2],  # e-stage4 1/16 channel = 64
            [6, 96, 3, 1],                 # channel = 96
            [6, 160, 3, 2], # e-stage5 1/32   channel = 160
            [6, 320, 1, 1],                  #channel = 320
        ]
        
        """
        backbone of MobileNetV2
        """
        features = []
        # conv1 layer
        features.append(conv_bn_relu(in_channel=3, out_channel=32, kernel_size=3, stride=2)) # e-stage0 /2
        # building inverted residual blocks
        input_channel = 32
        for t, c, n, s in inverted_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, t))
                input_channel = output_channel
        # # build 1×1 conv
        # features.append(conv_bn_relu(in_channel=input_channel,out_channel=1280,kernel_size=1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # -----decoder---------
        self.decoder1 = DecoderBlock(320,96)
        self.decoder2 = DecoderBlock(96,32)
        self.decoder3 = DecoderBlock(32,24)
        self.decoder4 = DecoderBlock(24,16)
        self.decoder5 = DecoderBlock(16,8)

        # -----output---------
        self.mask_head = nn.Conv2d(8, n_class, kernel_size = 1,bias=False)
        self.boundry_head = nn.Conv2d(8, 2, kernel_size = 1,bias=False)

        # initialize_weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = make_bilinear_weights(m.kernel_size[0], m.out_channels)  # same as caffe
                m.weight.data.copy_(initial_weight)
        

    def forward(self, x):
        # -----encoder---------
        # E-stage10  c = 32 size = 112
        # x = self.conv1(x) # c = 32

        # E-stage1  c = 16 size = 112
        for n in range(0,2):
            x = self.features[n](x)
        x1 = x

        # E-stage2  c = 24 size = 56
        for n in range(2,4):
            x = self.features[n](x)
        x2 = x

        # E-stage3  # c = 32 size = 28
        for n in range(4,7):
            x = self.features[n](x)
        x3 = x

        # E-stage4  c = 96 size = 14
        for n in range(7,14):
            x = self.features[n](x)
        x4 = x

        # E-stage5  c = 320 size = 7
        for n in range(14,18):
            x = self.features[n](x)
        x5 = x

        # -----decoder---------
        up1 = self.decoder1(x5) # 14×14×96
        up2 = self.decoder2(x4 + up1) #28×28×32
        up3 = self.decoder3(x3 + up2) #56×56×24
        up4 = self.decoder4(x2 + up3) #112×112×16
        up5 = self.decoder5(x1 + up4) #224×224×8

        # -----output---------
        mask = self.mask_head(up5)
        boundry = self.boundry_head(up5)

        return mask, boundry



