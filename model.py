import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init_normal(m):
    classname = m.__class__.__name__ # 부모가 아닌 현재 클래스명을 상속(다시)
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None: # bias 여부
            # hasattr(obj, name) : obj의 attribute에 name 존재 여부
            torch.nn.init.normal_(m.bias.data, 0.0)

    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

####################################
#       Generator
####################################
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.conv2d_layer = nn.Sequential(nn.ReflectionPad2d(1),
                                           nn.Conv2d(in_channels, in_channels, 3),
                                           nn.InstanceNorm2d(in_channels)
                                           )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv2d_layer(input) # in_channel = out_channel, kernel_size = 3
        output = self.relu(output)
        output = self.conv2d_layer(output)

        return output + input

class DownSample_Generator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(DownSample_Generator, self).__init__()

        self.convLayer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                       nn.InstanceNorm2d(out_channels)
                                       )

    def forward(self, input):
        input = self.convLayer(input)
        return input

class Upsampling_Generator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(Upsampling_Generator, self).__init__()

        self.convLayer = nn.Sequential(nn.Upsample(scale_factor=2),
                                       nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                       nn.InstanceNorm2d(out_channels)
                                       )

    def forward(self, input):
        input = self.convLayer(input)
        return input

class GeneratorResNet(nn.Module): # 이미지 사진 [channel, width ,height]
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0]

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        # Initial Convolution block
        self.first_convlayer = nn.Sequential(nn.ReflectionPad2d(channels),
                                   nn.Conv2d(channels, 64, 7),
                                   nn.InstanceNorm2d(64),
                                   )

        # DownSampling
        self.downsample1 = downSample_Generator(in_channels=64,
                                                out_channels=128,
                                                kernel_size=3,
                                                stride=2,
                                                padding=1)

        self.downsample2 = downSample_Generator(in_channels=128,
                                                out_channels=256,
                                                kernel_size=3,
                                                stride=2,
                                                padding=1)

        # Residual blcoks
        for i in range(1, num_residual_blocks + 1):
            locals()['residualLayer{}'.format(i)] = ResidualBlock(256)

        # Upsample
        self.upsample1 = Upsampling_Generator(in_channels=256,
                                              out_channels=128,
                                              stride=1,
                                              padding=1
                                              )
        self.upsample2 = Upsampling_Generator(in_channels=128,
                                              out_channels=64,
                                              stride=1,
                                              padding=1
                                              )

        self.last_convlayer = nn.Sequential(nn.ReflectionPad2d(channels),
                                       nn.Conv2d(64, channels, 7),
                                       )

    def forward(self, input): # input: [1, 3, 256, 256]
        input = self.first_convlayer(input)
        input = self.relu(input)

        #DownSample Layer
        input = self.downsample1(input)
        input = self.relu(input)
        input = self.downsample2(input)
        input = self.relu(input)

        # Residual Layer TODO 6BLOCK 9BLCOK일 때 방법

        # Upsample Layer
        input = self.upsample1(input)
        input = self.relu(input)
        input = self.upsample2(input)
        input = self.relu(input)

        # last convolution layer
        input = self.last_convlayer(input)
        input = self.tanh(input)

        return input













