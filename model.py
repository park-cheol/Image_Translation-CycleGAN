import torch
import torch.nn as nn
import torch.nn.functional as F
####################################
#       Generator
####################################
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        # 동일한 Sequential이 들어가기에 conv2d_layer를 하나만 선언하고 forward 사용
        # 이는 잘못된 방법
        self.conv2d_layer1 = nn.Sequential(nn.ReflectionPad2d(1),
                                           nn.Conv2d(in_channels, in_channels, 3),
                                           nn.InstanceNorm2d(in_channels),
                                           )

        self.conv2d_layer2 = nn.Sequential(nn.ReflectionPad2d(1),
                                          nn.Conv2d(in_channels, in_channels, 3),
                                          nn.InstanceNorm2d(in_channels),
                                          )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv2d_layer1(input) # in_channel = out_channel, kernel_size = 3
        output = self.relu(output)
        output = self.conv2d_layer2(output)
        return input + output

class DownSample_Generator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DownSample_Generator, self).__init__()

        self.convLayer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                       nn.InstanceNorm2d(out_channels)
                                       )

    def forward(self, input):
        output = self.convLayer(input)
        return output

class Upsampling_Generator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Upsampling_Generator, self).__init__()

        self.convLayer = nn.Sequential(nn.Upsample(scale_factor=2),
                                       nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                       nn.InstanceNorm2d(out_channels)
                                       )

    def forward(self, input):
        output = self.convLayer(input)
        return output

class GeneratorResNet(nn.Module): # 이미지 사진 [channel, width ,height]
    def __init__(self, input_shape, num_residual_blocks=9):
        # 우선 그냥 9로 코드 구현
        # input_shpae = list
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
        self.downsample1 = DownSample_Generator(in_channels=64,
                                                out_channels=128,
                                                kernel_size=3,
                                                stride=2,
                                                padding=1)

        self.downsample2 = DownSample_Generator(in_channels=128,
                                                out_channels=256,
                                                kernel_size=3,
                                                stride=2,
                                                padding=1)

        # Residual blcoks
        # 에러) 일단 주석처리
        #for i in range(1, self.num_residual_blocks + 1):
        #    locals()['self.residualLayer{}'.format(i)] = ResidualBlock(256)

        ###### 잘못된 코드인지 확인
        # self.residualLayer = ResidualBlock(256)
        self.residualLayer1 = ResidualBlock(256)
        self.residualLayer2 = ResidualBlock(256)
        self.residualLayer3 = ResidualBlock(256)
        self.residualLayer4 = ResidualBlock(256)
        self.residualLayer5 = ResidualBlock(256)
        self.residualLayer6 = ResidualBlock(256)
        self.residualLayer7 = ResidualBlock(256)
        self.residualLayer8 = ResidualBlock(256)
        self.residualLayer9 = ResidualBlock(256)

        # Upsample
        self.upsample1 = Upsampling_Generator(in_channels=256,
                                              out_channels=128,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1
                                              )
        self.upsample2 = Upsampling_Generator(in_channels=128,
                                              out_channels=64,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1
                                              )

        self.last_convlayer = nn.Sequential(nn.ReflectionPad2d(channels),
                                       nn.Conv2d(64, channels, 7),
                                       )

    def forward(self, input): # input: [1, 3, 256, 256]
        first_convLayer = self.first_convlayer(input)
        first_convLayer_relu = self.relu(first_convLayer)

        #DownSample Layer
        downsample_1 = self.downsample1(first_convLayer_relu)
        downsample_1_relu = self.relu(downsample_1)
        downsample_2 = self.downsample2(downsample_1_relu)
        downsample_2_relu = self.relu(downsample_2)

        # Residual Layer
        # 6은 일단 주석처리
        #if .num_residual_blocks == 6:
        #    residualLayer1 = self.residualLayer(downsample_2_relu)
        #    residualLayer2 = self.residualLayer2(residualLayer1)
        #    residualLayer3 = self.residualLayer3(residualLayer2)
        #    residualLayer4 = self.residualLayer4(residualLayer3)
        #    residualLayer5 = self.residualLayer5(residualLayer4)
        #    last_residualLayer = self.residualLayer6(residualLayer5)

        residualLayer1 = self.residualLayer1(downsample_2_relu)
        residualLayer2 = self.residualLayer2(residualLayer1)
        residualLayer3 = self.residualLayer3(residualLayer2)
        residualLayer4 = self.residualLayer4(residualLayer3)
        residualLayer5 = self.residualLayer5(residualLayer4)
        residualLayer6 = self.residualLayer6(residualLayer5)
        residualLayer7 = self.residualLayer7(residualLayer6)
        residualLayer8 = self.residualLayer8(residualLayer7)
        last_residualLayer = self.residualLayer9(residualLayer8)



        # Upsample Layer
        upsample1 = self.upsample1(last_residualLayer)
        upsample1_relu = self.relu(upsample1)
        upsample2 = self.upsample2(upsample1_relu)
        upsample2_relu = self.relu(upsample2)

        # last convolution layer
        last_convLayer = self.last_convlayer(upsample2_relu)
        output = self.tanh(last_convLayer)

        return output # [1, 3, 256, 256]

##############################
#       Discriminator
##############################
class Discriminator(nn.Module):
    # 논문에서 70 x 70 patchGan 사용
    # leakyrelu with a slope of 0.2
    # 1-dimensional output
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        # input_shape (3, 256, 256)
        channels, height, width = input_shape
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        # initial conv
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.zeropad = nn.ZeroPad2d((1, 0, 1, 0))

        self.convLayer_1 = nn.Sequential(nn.Conv2d(channels, 64, 4, stride=2, padding=1))

        self.convLayer_2 = nn.Sequential(nn.Conv2d(64, 128, 4, stride=2, padding=1),
                                         nn.InstanceNorm2d(128))

        self.convLayer_3 = nn.Sequential(nn.Conv2d(128, 256, 4, stride=2, padding=1),
                                         nn.InstanceNorm2d(256))

        self.convLayer_4 = nn.Sequential(nn.Conv2d(256, 512, 4, stride=2, padding=1),
                                         nn.InstanceNorm2d(512))

        self.convLayer_5 = nn.Conv2d(512, 1, 4, padding=1)

    def forward(self, input): # input[1, 3, 256, 256]
        convLayer_1 = self.convLayer_1(input)
        convLayer_1_relu = self.leakyrelu(convLayer_1)

        convLayer_2 = self.convLayer_2(convLayer_1_relu)
        convLayer_2_relu = self.leakyrelu(convLayer_2)

        convLayer_3 = self.convLayer_3(convLayer_2_relu)
        convLayer_3_relu = self.leakyrelu(convLayer_3)

        convLayer_4 = self.convLayer_4(convLayer_3_relu)
        convLayer_4_relu = self.leakyrelu(convLayer_4)

        zeropad = self.zeropad(convLayer_4_relu)
        output = self.convLayer_5(zeropad)

        return output # [1, 1, 16, 16]
# 참고 torch.manual_seed가 난수 생성 메서드 하나당 하나씩 만들어야함
# EX) torch.manual_seed(seed) -> randn -> 또 manual_seed -> randn 되더라



