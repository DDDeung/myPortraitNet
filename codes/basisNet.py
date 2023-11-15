import torch
import torch.nn as nn

class conv_bn_relu(nn.Sequential):
   def __init__(self, in_channel, out_channel, kernel_size = 3, stride = 1, groups = 1):
        """
        if kernel size = 3, padding = 1
        if kernel size = 1, padding  = 0
        """
        padding = (kernel_size - 1 ) // 2
        """
           groups  = 1:  classical convolution 
           groups  = in_channel:  deepwise separable convolution
           """
        super(conv_bn_relu, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size = kernel_size,stride=stride, padding=padding, groups=groups,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

# inverted residual block of MobileNetV2
class inverted_res_block(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expansion_rate):
        super(inverted_res_block, self).__init__()
        hidden_channel = in_channel * expansion_rate
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expansion_rate != 1: 
            # 1×1 pointwise conv
            layers.append(conv_bn_relu(in_channel, hidden_channel, kernel_size=1))
        layers.extend(
            # 3×3 depthwise conv 
            [conv_bn_relu(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1×1 pointwise conv(linear - no activate function) 
            nn.Conv2d(hidden_channel, out_channel,kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut: # whether use res
            return x + self.conv(x)
        else:
            return self.conv(x)



class d_block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(d_block, self).__init__()

        self.res_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.main_layers = nn.Sequential(
            # 3×3 depthwise conv 
            conv_bn_relu(in_channel,in_channel,kernel_size=3,stride=stride,groups=in_channel),
            # 1×1 pointwise conv
            conv_bn_relu(in_channel, out_channel, kernel_size=1),
            # 3×3 depthwise conv 
            conv_bn_relu(out_channel,out_channel,kernel_size=3,stride=stride,groups=out_channel),
            # 1×1 pointwise conv
            nn.Conv2d(out_channel, out_channel,kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.relu = nn.ReLU6(inplace=True)

        # if in_channel == out_channel: # whether use res
        #     self.use_shortcut = True

    def forward(self, x):
        out1 = self.res_conv(x)
        out2 = self.main_layers(x)
        return self.relu(out1+out2)
    

