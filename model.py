import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter, Softmax


class SpectralFeatureExtractor(nn.Module):
    
    def __init__(self, in_channels, out_channels, B):
        super(SpectralFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(12, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(12, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
        )
        # spec
        self.conv_ = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 7), padding=(0, 0, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 7), padding=(0, 0, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU()
        )

        self.conv_3 = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=(1, 1, 7), padding=(0, 0, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
        )
        self.fusion_conv = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 1))
        self.final_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, B)),
            nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (1, 1)),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
        )
        self.conv2d2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (1, 1)),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
        )

    def forward(self, x):
        identity = x
        spa_spec = self.conv(x)
        res1 = spa_spec
        x = self.conv2(spa_spec)
        spa_spec = self.conv3(torch.cat([res1, x], dim=1))
        spec = self.conv_(identity)
        res2 = self.conv_2(spec)
        spec = self.conv_3(torch.cat([res2, spec], dim=1))

        combined_feature = spa_spec + spec
        fusion_feature = self.fusion_conv(combined_feature)
        out = self.final_conv(fusion_feature)
        x = torch.squeeze(out)
        x = self.conv2d1(x)
        x_res = self.conv2d2(x)
        out = x + x_res
        return out


class SpectralChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, r):
        super(SpectralChannelAttention, self).__init__()
        self.channel_mapping_conv = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        max_pooled = self.max_pooling(x)
        avg_pooled = self.avg_pooling(x)
        pooled_features = max_pooled + avg_pooled
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        weights = self.mlp(pooled_features)
        reshaped_weights = weights.view(weights.size(0), weights.size(1), 1, 1)
        weighted_input = x * reshaped_weights
        out = x + weighted_input
        return out


class Spa(nn.Module):
    def __init__(self, out_channels, B):
        super(Spa, self).__init__()

        self.conv_3x3 = nn.Sequential(
            nn.Conv3d(1, out_channels, (3, 3, B), stride=1, padding=(1, 1, 0)),
            nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv_1x1 = nn.Sequential(
            nn.Conv3d(1, out_channels, (1, 1, B), stride=1, padding=0),
            nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels * 4, (1, 1)),
            nn.BatchNorm2d(out_channels * 4, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, (1, 1)),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (1, 1)),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        x_3x3 = self.conv_3x3(x)
        x_1x1 = self.conv_1x1(x)
        x = torch.cat([x_3x3, x_1x1], dim=1)
        x = torch.squeeze(x)
        x = self.conv2(self.conv1(x))
        x_res = self.conv3(x)
        x = x + x_res
        return x


class Lidar(nn.Module):
    def __init__(self, out_channels):
        super(Lidar, self).__init__()

        self.conv_3x3 = nn.Sequential(
            nn.Conv3d(1, out_channels, (3, 3, 1), stride=1, padding=(1, 1, 0)),
            nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv_1x1 = nn.Sequential(
            nn.Conv3d(1, out_channels, (1, 1, 1), stride=1, padding=0),
            nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels * 4, (1, 1)),
            nn.BatchNorm2d(out_channels * 4, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, (1, 1)),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (1, 1)),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        x_3x3 = self.conv_3x3(x)
        x_1x1 = self.conv_1x1(x)
        x = torch.cat([x_3x3, x_1x1], dim=1)
        x = torch.squeeze(x)
        x = self.conv2(self.conv1(x))
        x_res = self.conv3(x)
        x = x + x_res
        return x


class CAM(nn.Module):
    def __init__(self, in_dim):
        super(CAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.theno = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
        self.dim_reduce = nn.Sequential(
            nn.Conv1d(
                in_channels=in_dim,
                out_channels=1,
                kernel_size=1,
                bias=False,
            )
        )

    def forward(self, x1, x2):
        spec_batchsize, spec_C, height, width = x1.size()
        Fh_spec_a = x1.view(spec_batchsize, spec_C, -1)
        Fh_spec_b = x1.view(spec_batchsize, spec_C, -1).permute(0, 2, 1)  
        energy = torch.bmm(Fh_spec_a, Fh_spec_b)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        spec_attention = self.softmax(energy_new)
        Fh_spec_c = x1.view(spec_batchsize, spec_C, -1)
        out1 = torch.bmm(spec_attention, Fh_spec_c)
        out1 = out1.view(spec_batchsize, spec_C, height, width)
        out1 = self.gamma * out1 + x1
        l_batchsize, lidar_C, height, width = x2.size()
        Fl_b = x2.view(l_batchsize, lidar_C, -1).permute(0, 2, 1)

        A_M = torch.matmul(Fh_spec_a, Fl_b)
        A_M = self.dim_reduce(A_M)
        out2 = A_M.view(spec_batchsize, x1.size(1), 1, 1)
        out = out1 + out2 * x2
        return out


class ConvUnit_NP(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(ConvUnit_NP, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, bias=True)
        self.bn = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.bn(self.conv(x)))
        return x


class Classification_Module(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(Classification_Module, self).__init__()
        self.conv1 = ConvUnit_NP(input_channels, 256)
        self.conv2 = ConvUnit_NP(256, 256)
        self.conv3 = ConvUnit_NP(256, 256)
        self.conv4 = ConvUnit_NP(256, 256)
        self.conv5 = ConvUnit_NP(256, 256)
        self.conv6 = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = torch.squeeze(out)
        return out


class MSSCENet(nn.Module):
    def __init__(self, num_classes):
        super(MSSCENet, self).__init__()
        self.spec = SpectralFeatureExtractor(in_channels=1, out_channels=12, B=b)
        self.spec_atten = SpectralChannelAttention(in_channels=12, out_channels=12, r=2)
        self.spa = Spa(out_channels=12, B=b)
        self.lidar = Lidar(out_channels=12)
        # self.pam = SpaLidar(12)
        self.cam = CAM(12)
        # self.SPFSM = SPFSMModule()
        # self.pix = GetSpatialFeatures()
        self.classification = Classification_Module(c, num_classes)

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1).permute(0, 1, 3, 4, 2)
        x2 = x2.unsqueeze(1).permute(0, 1, 3, 4, 2)
        out_spa = self.spa(x1)

        out_spec = self.spec(x1)
        out_spec = self.spec_atten(out_spec)
        out_lidar = self.lidar(x2)
        out1 = self.cam(out_spec, out_lidar)
        # out2 = self.pam(out_spa, out_lidar)
        # center_pixel, neighboring_pixel = self.pix(out_spa)
        # out3 = self.SPFSM(center_pixel, neighboring_pixel) + out_spa
        out3 = out3 + out_spec
        # out = self.classification(torch.cat([out1, out2, out3], dim=1))
        # return out
        return out3