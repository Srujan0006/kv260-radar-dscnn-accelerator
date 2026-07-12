# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class Radar2DCNN(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(Radar2DCNN, self).__init__()
        self.module_0 = py_nndct.nn.Input() #Radar2DCNN::input_0(Radar2DCNN::nndct_input_0)
        self.module_1 = py_nndct.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Radar2DCNN::Radar2DCNN/Sequential[features]/ConvBNReLU[0]/Sequential[block]/Conv2d[0]/ret.3(Radar2DCNN::nndct_conv2d_1)
        self.module_2 = py_nndct.nn.ReLU(inplace=False) #Radar2DCNN::Radar2DCNN/Sequential[features]/ConvBNReLU[0]/Sequential[block]/ReLU[2]/ret.7(Radar2DCNN::nndct_relu_2)
        self.module_3 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #Radar2DCNN::Radar2DCNN/Sequential[features]/MaxPool2d[1]/1169(Radar2DCNN::nndct_maxpool_3)
        self.module_4 = py_nndct.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Radar2DCNN::Radar2DCNN/Sequential[features]/ConvBNReLU[2]/Sequential[block]/Conv2d[0]/ret.9(Radar2DCNN::nndct_conv2d_4)
        self.module_5 = py_nndct.nn.ReLU(inplace=False) #Radar2DCNN::Radar2DCNN/Sequential[features]/ConvBNReLU[2]/Sequential[block]/ReLU[2]/ret.13(Radar2DCNN::nndct_relu_5)
        self.module_6 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #Radar2DCNN::Radar2DCNN/Sequential[features]/MaxPool2d[3]/1214(Radar2DCNN::nndct_maxpool_6)
        self.module_7 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Radar2DCNN::Radar2DCNN/Sequential[features]/ConvBNReLU[4]/Sequential[block]/Conv2d[0]/ret.15(Radar2DCNN::nndct_conv2d_7)
        self.module_8 = py_nndct.nn.ReLU(inplace=False) #Radar2DCNN::Radar2DCNN/Sequential[features]/ConvBNReLU[4]/Sequential[block]/ReLU[2]/ret.19(Radar2DCNN::nndct_relu_8)
        self.module_9 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #Radar2DCNN::Radar2DCNN/Sequential[features]/MaxPool2d[5]/1259(Radar2DCNN::nndct_maxpool_9)
        self.module_10 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Radar2DCNN::Radar2DCNN/Sequential[features]/ConvBNReLU[6]/Sequential[block]/Conv2d[0]/ret.21(Radar2DCNN::nndct_conv2d_10)
        self.module_11 = py_nndct.nn.ReLU(inplace=False) #Radar2DCNN::Radar2DCNN/Sequential[features]/ConvBNReLU[6]/Sequential[block]/ReLU[2]/ret.25(Radar2DCNN::nndct_relu_11)
        self.module_12 = py_nndct.nn.AvgPool2d(kernel_size=[12, 12], stride=[12, 12], padding=[0, 0], ceil_mode=False, count_include_pad=True) #Radar2DCNN::Radar2DCNN/Sequential[features]/AvgPool2d[7]/1303(Radar2DCNN::nndct_avgpool_12)
        self.module_13 = py_nndct.nn.Conv2d(in_channels=64, out_channels=4, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Radar2DCNN::Radar2DCNN/Conv2d[classifier_conv]/ret.27(Radar2DCNN::nndct_conv2d_13)
        self.module_14 = py_nndct.nn.Module('nndct_flatten') #Radar2DCNN::Radar2DCNN/ret(Radar2DCNN::nndct_flatten_14)

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_0 = self.module_6(output_module_0)
        output_module_0 = self.module_7(output_module_0)
        output_module_0 = self.module_8(output_module_0)
        output_module_0 = self.module_9(output_module_0)
        output_module_0 = self.module_10(output_module_0)
        output_module_0 = self.module_11(output_module_0)
        output_module_0 = self.module_12(output_module_0)
        output_module_0 = self.module_13(output_module_0)
        output_module_0 = self.module_14(input=output_module_0, start_dim=1, end_dim=-1)
        return output_module_0
