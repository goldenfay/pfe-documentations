import torch.nn as nn
import torch
from torchvision import models
    # User's modules
import model
from model import *

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
    # User's module from another directory
sys.path.append(os.path.join(parentdir, "bases"))
from utils import BASE_PATH

class CSRNet(Model):
    def __init__(self,frontEnd,backEnd,output_layer, weightsFlag=False):
        super(CSRNet, self).__init__()
        # self.seen = 0
        # self.frontEnd_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        # self.backEnd_feat  = [512, 512, 512,256,128,64]
        self.frontEnd = frontEnd
        self.backEnd = backEnd
        self.output_layer = output_layer 
            # If the weights are not initialized, use the VGG16 architecture for frontEnd
        if not weightsFlag:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in xrange(len(self.frontEnd.state_dict().items())):
                self.frontEnd.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
                
    def forward(self,x):
        x = self.frontEnd(x)
        x = self.backEnd(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def default_architecture():
        frontEnd_shape={
            
                'C2D':{
                    'in_channels':3,
                    'out_channels': 64,
                    'ks': 3,
                    'stride': 1,
                    'padding': 1,
                    'dilation': 1
                },
                'R':{
                    'inplace':True
                },
                'C2D':{
                    'in_channels':3,
                    'out_channels': 64,
                    'ks': 3,
                    'stride': 1,
                    'padding': 1,
                    'dilation': 1
                },
                'R':{
                    'inplace':True
                },
                'M':{
                    'ks': 2,
                    'stride': 2,
                   
                },
                'C2D':{
                    'in_channels':3,
                    'out_channels': 256,
                    'ks': 3,
                    'stride': 1,
                    'padding': 1,
                    'dilation': 1
                },
                'R':{
                    'inplace':True
                },
                'C2D':{
                    'in_channels':3,
                    'out_channels': 256,
                    'ks': 3,
                    'stride': 1,
                    'padding': 1,
                    'dilation': 1
                },
                'R':{
                    'inplace':True
                },
                'C2D':{
                    'in_channels':3,
                    'out_channels': 256,
                    'ks': 3,
                    'stride': 1,
                    'padding': 1,
                    'dilation': 1
                },
                'M':{
                    'ks': 2,
                    'stride': 2,
                   
                },
                'C2D':{
                    'in_channels':3,
                    'out_channels': 512,
                    'ks': 3,
                    'stride': 1,
                    'padding': 1,
                    'dilation': 1
                },
                'R':{
                    'inplace':True
                },
                'C2D':{
                    'in_channels':3,
                    'out_channels': 512,
                    'ks': 3,
                    'stride': 1,
                    'padding': 1,
                    'dilation': 1
                },
                'R':{
                    'inplace':True
                },
                'C2D':{
                    'in_channels':3,
                    'out_channels': 512,
                    'ks': 3,
                    'stride': 1,
                    'padding': 1,
                    'dilation': 1
                },
                'R':{
                    'inplace':True
                }

            
        }
        backEnd_shape={
                'C2D':{
                        'in_channels':512,
                        'out_channels': 512,
                        'ks': 3,
                        'stride': 1,
                        'padding': 2,
                        'dilation': 2
                    },
                'R':{
                    'inplace':True
                },
                'C2D':{
                        'in_channels':512,
                        'out_channels': 512,
                        'ks': 3,
                        'stride': 1,
                        'padding': 2,
                        'dilation': 2
                    },
                'R':{
                    'inplace':True
                },
                'C2D':{
                        'in_channels':512,
                        'out_channels': 512,
                        'ks': 3,
                        'stride': 1,
                        'padding': 2,
                        'dilation': 2
                    },
                'R':{
                    'inplace':True
                },
                'C2D':{
                        'in_channels':512,
                        'out_channels': 256,
                        'ks': 3,
                        'stride': 1,
                        'padding': 2,
                        'dilation': 2
                    },
                'R':{
                    'inplace':True
                },
                'C2D':{
                        'in_channels':512,
                        'out_channels': 128,
                        'ks': 3,
                        'stride': 1,
                        'padding': 2,
                        'dilation': 2
                    },
                'R':{
                    'inplace':True
                },
                'C2D':{
                        'in_channels':512,
                        'out_channels': 64,
                        'ks': 3,
                        'stride': 1,
                        'padding': 2,
                        'dilation': 2
                    },
                'R':{
                    'inplace':True
                }  
        }
        output_layer=nn.Conv2d(64, 1, kernel_size=1)

        return frontEnd_shape,backEnd_shape,output_layer         
            
                