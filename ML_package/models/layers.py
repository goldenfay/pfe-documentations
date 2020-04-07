import torch
from torch import nn




def conv2D_layer(params:dict):
    return nn.Conv2d(params['in_channels'],
                    params['out_channels'],
                    kernel_size=params['ks'],
                    stride=params['stride'],
                    padding=params['padding'],
                    dilation=params['dilation']
                    )
def relu_layer(params:dict):
    return nn.ReLU(inplace=params['inplace'])

def maxpool_layer(params:dict):
    return nn.MaxPool2d(params['ks'],stride=params['stride'])

def fc_layer(params:dict):
    return nn.Linear(params['in_size'],params['out_size'])
    


def construct_net(schema:list,weight_flag=False):
    arch=[]
    for layer_key,params in schema:
        if layer_key=='M':
            arch+=[maxpool_layer(params)]
        elif layer_key=='C2D':
            arch+=[conv2D_layer(params)]
        elif layer_key=='R':
            arch+=[relu_layer(params)]
        elif layer_key=='BR':
            arch+=[nn.BatchNorm2d(params['out_channels'])]    
            arch+=[relu_layer(params)]    
        elif layer_key=='FC':
            arch+=[fc_layer(params)]    

    print(arch)
    return nn.Sequential(*arch)        


