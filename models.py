# ours, based on spsg

import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def count_num_model_params(model):
    num = 0
    for p in list(model.parameters()):
        cur = 1
        for s in list(p.size()):
            cur = cur * s
        num += cur
    return num

class GeoEncoder(nn.Module):
    def __init__(self, nf_in_geo, nf_in_color, nf, max_data_size, truncation):
        self.nf = nf
        self.input_mask = nf_in_color > 3
        self.use_bias = True
        self.truncation = truncation
        self.interpolate_mode = 'nearest'

        # === geo net === 
        self.geo_0 = nn.Sequential(
                nn.Conv2d(nf_in_geo, self.nf//2, 5, 1, 2, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(self.nf//2),
                nn.Conv2d(self.nf//2, self.nf, 4, 2, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(self.nf),
                nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(self.nf)
        )
        self.geo_1 = nn.Sequential(
                nn.Conv2d(self.nf, 2*self.nf, 4, 2, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(2*self.nf),
                # 1
                nn.Conv2d(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(2*self.nf),
                # 2
                nn.Conv2d(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(2*self.nf),
                # 3
                nn.Conv2d(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(2*self.nf),
                # 4
                nn.Conv2d(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(2*self.nf),
                # 5
                nn.Conv2d(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(2*self.nf),
                # 6
                nn.Conv2d(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(2*self.nf),
                # 7
                nn.Conv2d(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(2*self.nf),
                # 8
                nn.Conv2d(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(2*self.nf),
        )
        self.geo_2 = torch.nn.Sequential(
                torch.nn.Conv2d(2*self.nf, self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(self.nf),
                torch.nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(self.nf)
        )
        self.coarse_0_geo = nn.Sequential(
            nn.Conv2d(self.nf, self.nf, 4, 2, 1, bias=self.use_bias),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(self.nf),
        )

        num_params_geo = count_num_model_params(self.geo_0) + count_num_model_params(self.geo_1) + count_num_model_params(self.geo_2) + count_num_model_params(self.coarse_0_geo)
        print('#params(geo) = ', num_params_geo)

    def forward(self, x, mask):        
        if self.input_mask:
            x = torch.cat([x, mask], 1)
            x_geo = x[:,:1,:,:]
            mask = x[:,4:,:,:]
        else:
            x_geo = x[:,:1,:,:]
        x_geo[torch.abs(x_geo) >= self.truncation-0.01] = 0
        
        scale_factor = 2
        # down sample
        x_geo = self.geo_0(x_geo)
        x_geo = self.geo_1(x_geo)
        # H0/4, W0/4
        # up sample
        x_geo = torch.nn.functional.interpolate(x_geo, scale_factor=scale_factor, mode=self.interpolate_mode)
        x_geo = self.geo_2(x_geo)
        x_geo = torch.nn.functional.interpolate(x_geo, scale_factor=scale_factor, mode=self.interpolate_mode)
        # H0,W0
        x_geo = self.coarse_0_geo(x_geo)
        # H0/2,W0/2

        return x_geo

class GeoVisEncoder(nn.Module):
    def __init__(self, nf_in_geo, nf_in_color, nf, max_data_size, truncation):
        self.nf = nf
        self.input_mask = nf_in_color > 3
        self.use_bias = True
        self.truncation = truncation
        self.interpolate_mode = 'nearest'

        # === geo net === 
        self.geo_0 = nn.Sequential(
                nn.Conv2d(nf_in_geo, self.nf//2, 5, 1, 2, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(self.nf//2),
                nn.Conv2d(self.nf//2, self.nf, 4, 2, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(self.nf),
                nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(self.nf)
        )
        self.geo_1 = nn.Sequential(
                nn.Conv2d(self.nf, 2*self.nf, 4, 2, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(2*self.nf),
                # 1
                nn.Conv2d(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(2*self.nf),
                # 2
                nn.Conv2d(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(2*self.nf),
                # 3
                nn.Conv2d(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(2*self.nf),
                # 4
                nn.Conv2d(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(2*self.nf),
                # 5
                nn.Conv2d(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(2*self.nf),
                # 6
                nn.Conv2d(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(2*self.nf),
                # 7
                nn.Conv2d(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(2*self.nf),
                # 8
                nn.Conv2d(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(2*self.nf),
        )
        self.geo_2 = torch.nn.Sequential(
                torch.nn.Conv2d(2*self.nf, self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(self.nf),
                torch.nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(self.nf)
        )
        self.coarse_0_geo = nn.Sequential(
                nn.Conv2d(self.nf, self.nf, 4, 2, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(self.nf),
        )
        # === coarse net === 
        self.coarse_0 = nn.Sequential(
                nn.Conv2d(nf_in_color, self.nf, 5, 1, 2, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(self.nf),
                nn.Conv2d(self.nf, 2*self.nf, 4, 2, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(2*self.nf),
                nn.Conv2d(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(2*self.nf)
        )
        nf1 = 3*self.nf
        nf_factor = 4
        self.coarse_1 = nn.Sequential(
                nn.Conv2d(nf1, nf_factor*self.nf, 4, 2, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(nf_factor*self.nf),
                # 1
                nn.Conv2d(nf_factor*self.nf, nf_factor*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(nf_factor*self.nf),
                # 2
                nn.Conv2d(nf_factor*self.nf, nf_factor*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(nf_factor*self.nf),
                # 3
                nn.Conv2d(nf_factor*self.nf, nf_factor*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(nf_factor*self.nf),
                # 4
                nn.Conv2d(nf_factor*self.nf, nf_factor*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(nf_factor*self.nf),
                # 5
                nn.Conv2d(nf_factor*self.nf, nf_factor*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(nf_factor*self.nf),
                # 6
                nn.Conv2d(nf_factor*self.nf, nf_factor*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(nf_factor*self.nf),
                # 7
                nn.Conv2d(nf_factor*self.nf, nf_factor*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(nf_factor*self.nf),
                # 8
                nn.Conv2d(nf_factor*self.nf, nf_factor*self.nf, 3, 1, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(nf_factor*self.nf),
        )

        num_params_geo = count_num_model_params(self.geo_0) + count_num_model_params(self.geo_1) + count_num_model_params(self.geo_2) + count_num_model_params(self.coarse_0_geo)
        print('#params(geo) = ', num_params_geo)
        num_params_coarse = count_num_model_params(self.coarse_0) + count_num_model_params(self.coarse_1)
        print('#params(coarse) = ', num_params_coarse)

    def forward(self, x, mask):        
        if self.input_mask:
            x = torch.cat([x, mask], 1)
            x_geo = x[:,:1,:,:]
            mask = x[:,4:,:,:]
        else:
            x_geo = x[:,:1,:,:]
        x_geo[torch.abs(x_geo) >= self.truncation-0.01] = 0
        
        scale_factor = 2
        # down sample
        x_geo = self.geo_0(x_geo)
        x_geo = self.geo_1(x_geo)
        # H0/4, W0/4
        # up sample
        x_geo = torch.nn.functional.interpolate(x_geo, scale_factor=scale_factor, mode=self.interpolate_mode)
        x_geo = self.geo_2(x_geo)
        x_geo = torch.nn.functional.interpolate(x_geo, scale_factor=scale_factor, mode=self.interpolate_mode)
        # H0,W0
        x_geo = self.coarse_0_geo(x_geo)
        # H0/2,W0/2

        x_color = x[:,1:4,:,:]
        x_color = x_color*2-1
        if self.input_mask:
            masked_x = x_color * (1 - mask) + mask
            coarse_x = self.coarse_0(torch.cat((masked_x, mask), dim=1))
        else:
            coarse_x = self.coarse_0(x_color)
        coarse_x = torch.cat((coarse_x, x_geo), dim=1)
        # H0/2,W0/2
        coarse_x = self.coarse_1(coarse_x)
        # H0/4,W0/4

        return coarse_x