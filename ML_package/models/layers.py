import torch
from torch import nn




def conv2D_layer(params:dict):
    return nn.Conv2d(params['in_channels'],
                    params['out_channels'],
                    params['ks'],
                    params['stride'],
                    params['padding'],
                    params['dilation']
                    )
def relu_layer(params:dict):
    return nn.ReLU(inplace=params['inplace'])

def maxpool_layer(params:dict):
    return nn.MaxPool2d(params['ks'],params['stride'])

def fc_layer(params:dict):
    return nn.Linear(params['in_size'],params['out_size'])


def construct_net(schema:dict,weight_flag=False):
    arch=[]
    for layer_key in schema:
        if layer_key=='M':
            arch.append(maxpool_layer(schema[layer_key]))
        elif layer_key=='C2D':
            arch.append(conv2D_layer(schema[layer_key]))
        elif layer_key=='R':
            arch.append(relu_layer(schema[layer_key]))
        elif layer_key=='BR':
            arch.append(nn.BatchNorm2d(schema[layer_key]['out_channels']))    
            arch.append(relu_layer(schema[layer_key]))    
        elif layer_key=='FC':
            arch.append(fc_layer(schema[layer_key]))    


    return nn.Sequential(*arch)        


