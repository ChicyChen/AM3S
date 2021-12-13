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
        ## up sample
        ## x_geo = torch.nn.functional.interpolate(x_geo, scale_factor=scale_factor, mode=self.interpolate_mode)
        x_geo = self.geo_2(x_geo)
        # H0/4, W0/4
        ## x_geo = torch.nn.functional.interpolate(x_geo, scale_factor=scale_factor, mode=self.interpolate_mode)
        ## H0,W0
        x_geo = self.coarse_0_geo(x_geo)
        # H0/8,W0/8

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
        self.coarse_0_vis = nn.Sequential(
                nn.Conv2d(nf_factor*self.nf, nf_factor*self.nf, 4, 2, 1, bias=self.use_bias),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(self.nf),
        )

        num_params_geo = count_num_model_params(self.geo_0) + count_num_model_params(self.geo_1) + count_num_model_params(self.geo_2) + count_num_model_params(self.coarse_0_geo)
        print('#params(geo) = ', num_params_geo)
        num_params_coarse = count_num_model_params(self.coarse_0) + count_num_model_params(self.coarse_1) + count_num_model_params(self.coarse_0_vis)
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
        coarse_x = self.coarse_0_vis(coarse_x)
        # H0/8,W0/8

        return coarse_x


class CameraPredictor(nn.Module):
    '''
        Multilayer Perceptron.
    '''
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 784),
            nn.ReLU(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)


class SNConv2WithActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNConv2WithActivation, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        #self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.conv2d(x)
        if self.activation is not None:
            return self.activation(x)
        return x


class Discriminator2D(nn.Module):
    def __init__(self, nf_in, nf, patch_size, image_dims, patch, use_bias, disc_loss_type='vanilla'):
        nn.Module.__init__(self)
        self.use_bias = use_bias
        approx_receptive_field_sizes = [4, 10, 22, 46, 94, 190, 382, 766]
        num_layers = len(approx_receptive_field_sizes)
        if patch:
            for k in range(len(approx_receptive_field_sizes)):
                if patch_size < approx_receptive_field_sizes[k]:
                    num_layers = k
                    break
        assert(num_layers >= 1)
        self.patch = patch
        self.nf = nf
        dim = min(image_dims[0], image_dims[1])
        num = int(math.floor(math.log(dim, 2)))
        num_layers = min(num, num_layers)
        activation = None if num_layers == 1 else torch.nn.LeakyReLU(0.2, inplace=True)
        self.discriminator_net = torch.nn.Sequential(
            SNConv2WithActivation(nf_in, 2*nf, 4, 2, 1, activation=activation, bias=self.use_bias),
        )
        if num_layers > 1:
            activation = None if num_layers == 2 else torch.nn.LeakyReLU(0.2, inplace=True)
            self.discriminator_net.add_module('p1', SNConv2WithActivation(2*nf, 4*nf, 4, 2, 1, activation=activation, bias=self.use_bias))
        if num_layers > 2:
            activation = None if num_layers == 3 else torch.nn.LeakyReLU(0.2, inplace=True)
            self.discriminator_net.add_module('p2', SNConv2WithActivation(4*nf, 8*nf, 4, 2, 1, activation=activation, bias=self.use_bias))
        for k in range(3, num_layers):
            activation = None if num_layers == k+1 else torch.nn.LeakyReLU(0.2, inplace=True)
            self.discriminator_net.add_module('p%d' % k, SNConv2WithActivation(8*nf, 8*nf, 4, 2, 1, activation=activation, bias=self.use_bias))
        self.final = None
        if not patch or disc_loss_type != 'hinge': #hack
            self.final = torch.nn.Conv2d(nf*8, 1, 1, 1, 0)        
        num_params = count_num_model_params(self.discriminator_net)
        print('#params discriminator', count_num_model_params(self.discriminator_net))
        
        self.compute_valid = None
        if patch:
            self.compute_valid = torch.nn.Sequential(
                torch.nn.AvgPool2d(4, stride=2, padding=1),
            )
            for k in range(1, num_layers):
                self.compute_valid.add_module('p%d' % k, torch.nn.AvgPool2d(4, stride=2, padding=1))

    def forward(self, x, alpha=None):
        for k in range(len(self.discriminator_net)-1):
            x = self.discriminator_net[k](x)
        x = self.discriminator_net[-1](x) 
        
        if self.final is not None:
            x = self.final(x)
        x = torch.permute(x, (0, 2, 3, 1))
        return x