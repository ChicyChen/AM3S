import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from math import sqrt
import numpy as np
from pytorch_msssim import SSIM


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = []

    def forward(self, pred, orig, max_depth=10, smaller=False):
        target = orig #max_depth/orig

        if smaller:
            target = F.max_pool2d(input=target,kernel_size=4, stride=4, padding=0)
        print(target.shape)
        print(pred.shape)
        L_depth = nn.L1Loss()(pred, target)


        sobel_x = torch.tensor([[1., 0. , -1.], [2., 0., -2.], [1., 0. , -1.]]).expand(1,100,3,3)
        sobel_y = torch.tensor([[1., 2. , 1.], [0., 0., 0.], [-1., -2. , -1.]]).expand(1,100,3,3)
        grad_pred_x = F.conv2d(pred, weight=sobel_x, stride=1, padding=1)
        grad_pred_y = F.conv2d(pred, weight=sobel_y, stride=1, padding=1)
        grad_target_x = F.conv2d(target, weight=sobel_x, stride=1, padding=1)
        grad_target_y = F.conv2d(target, weight=sobel_y, stride=1, padding=1)

        L_grad = nn.L1Loss()(grad_pred_x, grad_target_x) + nn.L1Loss()(grad_pred_y, grad_target_y)

        SSIMFunc = SSIM()
        pred_ssim_in = torch.cat((pred, pred, pred), 1)
        target_ssim_in = torch.cat((target, target, target), 1)
        L_ssim = (1 - SSIMFunc(pred_ssim_in, target_ssim_in))/2
        self.ssim.append(L_ssim)
        total_L = 0.1 * L_depth + L_grad + L_ssim
        return total_L

    def get_ssim(self):
        return self.ssim
    



class Model(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.densenet_conv1_out = None
        self.densenet_pool1_out = None
        self.densenet_pool2_out = None
        self.densenet_pool3_out = None
        
        self.CONV0 = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.downsample = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)

        densenet = models.densenet169(pretrained=True)
        self.model = nn.Sequential(*list(densenet.children())[0][:-1])

        self.upsample_double = nn.Upsample(scale_factor=(2,2))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.CONV2 = nn.Conv2d(in_channels=1664, out_channels=1664, kernel_size=1, stride=1, padding=0)
        self.CONV3 = nn.Conv2d(in_channels=104, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

        self.UP1_CONV_A = nn.Conv2d(in_channels=1920, out_channels=832, kernel_size=3, stride=1, padding=1)
        self.UP1_CONV_B = nn.Conv2d(in_channels=832, out_channels=832, kernel_size=3, stride=1, padding=1)

        self.UP2_CONV_A = nn.Conv2d(in_channels=960, out_channels=416, kernel_size=3, stride=1, padding=1)
        self.UP2_CONV_B = nn.Conv2d(in_channels=416, out_channels=416, kernel_size=3, stride=1, padding=1)

        self.UP3_CONV_A = nn.Conv2d(in_channels=480, out_channels=208, kernel_size=3, stride=1, padding=1)
        self.UP3_CONV_B = nn.Conv2d(in_channels=208, out_channels=208, kernel_size=3, stride=1, padding=1)

        self.UP4_CONV_A = nn.Conv2d(in_channels=272, out_channels=104, kernel_size=3, stride=1, padding=1)
        self.UP4_CONV_B = nn.Conv2d(in_channels=104, out_channels=104, kernel_size=3, stride=1, padding=1)
        
        # Freezing model
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x, smaller=False):
        children_counter = 0
        in_counter = 0
        x = self.CONV0(x)
        if smaller:
            x = self.downsample(x)

        # Check output first to understand this
        for n, c in self.model.named_children():
            children_counter+=1
            if children_counter == 6:
                for n_in, c_in in c.named_children():
                    x = c_in(x)
                    in_counter += 1
                    if in_counter == 4:
                        self.densenet_pool2_out = x
                        #print("saved pool2")

            elif children_counter == 8:
                for n_in, c_in in c.named_children():
                    x = c_in(x)
                    in_counter += 1
                    if in_counter == 8:
                        self.densenet_pool3_out = x
                        #print("saved pool3")
            
            else:
                #print("children: ", n, " done")
                x = c(x)
                if children_counter == 1:
                    self.densenet_conv1_out = x
                if children_counter == 4:
                    self.densenet_pool1_out = x
          
        x = self.CONV2(x)
        x = self.upsample_double(x)
        if smaller:
            x = nn.ZeroPad2d(padding=(0,0,1,0))(x)
        x = torch.cat([x, self.densenet_pool3_out], dim=1)
        x = self.UP1_CONV_A(x)
        x = self.UP1_CONV_B(x)
        x = self.leaky_relu(x)

        x = self.upsample_double(x)
        if smaller:
            x = nn.ZeroPad2d(padding=(0,0,1,0))(x)
        x = torch.cat([x, self.densenet_pool2_out], dim=1)
        x = self.UP2_CONV_A(x)
        x = self.UP2_CONV_B(x)
        x = self.leaky_relu(x)
        
        x = self.upsample_double(x)
        x = torch.cat([x, self.densenet_pool1_out], dim=1)
        x = self.UP3_CONV_A(x)
        x = self.UP3_CONV_B(x)
        x = self.leaky_relu(x)

        x = self.upsample_double(x)
        x = torch.cat([x, self.densenet_conv1_out], dim=1)
        x = self.UP4_CONV_A(x)
        x = self.UP4_CONV_B(x)

        x = self.upsample_double(x)
        x = self.CONV3(x)
        
        return x