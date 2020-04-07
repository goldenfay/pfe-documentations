import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
import os,sys,inspect,glob,re,time,datetime
import numpy as np
    # User's modules
import model
from model import *
import layers

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
    # User's module from another directory
sys.path.append(os.path.join(parentdir, "bases"))
from utils import BASE_PATH


class CSRNet(Model):
    def __init__(self,frontEnd,backEnd,output_layer_arch, weightsFlag=False):
        super(CSRNet, self).__init__()
        # self.seen = 0
        # self.frontEnd_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        # self.backEnd_feat  = [512, 512, 512,256,128,64]
        self.frontEnd = layers.construct_net(frontEnd)
        self.backEnd = layers.construct_net(backEnd)
        self.output_layer = layers.construct_net(output_layer_arch) 
            # If the weights are not initialized, use the VGG16 architecture for frontEnd
        if not weightsFlag:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(self.frontEnd.state_dict().items())):
                self.frontEnd.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
        
    
    
    def __init__(self, weightsFlag=False):
        super(CSRNet, self).__init__() 
        self.frontEnd ,self.backEnd ,self.output_layer=self.default_architecture()
        self.frontEnd ,self.backEnd =layers.construct_net(self.frontEnd),layers.construct_net(self.backEnd)
        if not weightsFlag:
            mod = models.vgg16(pretrained = True)
            for i in range(23):
                print(mod.features[i],'\t',list(self.frontEnd.modules())[i])
            self._initialize_weights()
            self.frontEnd.load_state_dict(mod.features[0:len(list(self.frontEnd.modules()))].state_dict())

    def forward(self,x):
        x = self.frontEnd(x)
        print('After Front end',x.size())
        x = self.backEnd(x)
        print('After back end',x.size())
        x = self.output_layer(x)
        x = F.interpolate(x,scale_factor=8, mode='bilinear')
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
                    'in_channels':64,
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
                    'in_channels':64,
                    'out_channels': 128,
                    'ks': 3,
                    'stride': 1,
                    'padding': 1,
                    'dilation': 1
                },
                'R':{
                    'inplace':True
                },
                'C2D':{
                    'in_channels':128,
                    'out_channels': 128,
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
                    'in_channels':128,
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
                    'in_channels':256,
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
                    'in_channels':256,
                    'out_channels': 256,
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
                    'in_channels':256,
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
                    'in_channels':512,
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
                    'in_channels':512,
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
                        'in_channels':256,
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
                        'in_channels':128,
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

    def train_model(self,train_dataloader,test_dataloader,train_params:TrainParams,resume=False):
        '''
            Start training the model with specified parameters.
        '''
        print("####### Training The model...")
        self.params=train_params
        self.optimizer=train_params.optimizer
            # Get the device (GPU/CPU) and migrate the model to it
        device=train_params.device
        print("\t Setting up model on ",device.type,"...")    
        self.to(device)
        if not os.path.exists(os.path.join(utils.BASE_PATH,'checkpoints2')):
            os.mkdir(os.path.join(utils.BASE_PATH,'checkpoints2'))
           
        
            # Initialize training variables
        print("\t Initializing ","...")    
        self.min_MAE=10000
        self.min_epoch=0
        epochs_list=[]
        train_loss_list=[]
        test_error_list=[]
        start_epoch=0

            # If resume option is specified, restore state of model and resume training
        if resume:
            params_hist=[int(re.sub("[^0-9]+","",file_path[list(re.finditer("[\\\/]",file_path))[-1].start(0):])) for file_path in glob.glob(os.path.join(os.path.join(utils.BASE_PATH,'checkpoints2',self.__class__.__name__),'*.pth'))]
            
            

            if len(params_hist)>0:
                print("\t Restore Checkpoints2 found! Resuming training...")
                # start_epoch=int(re.sub("[^0-9]+","",params_hist[-1][list(re.finditer("[\\\/]",params_hist[-1]))[-1].start(0):]))
                start_epoch=max(sorted(params_hist))
                #start_epoch=435
                last_epoch=glob.glob(os.path.join(os.path.join(utils.BASE_PATH,'checkpoints2',self.__class__.__name__,'epoch_'+str(start_epoch)+'.pth')))[0]
               
                    #//////////
                _,self.min_MAE,self.min_epoch=self.load_chekpoint(last_epoch)    

        start_epoch+=1   

        start=time.time()        

            # Start Train
        for epoch in range(start_epoch,train_params.maxEpochs):
                # Set the Model on training mode
            self.train()
            epoch_loss=0
                # Run training pass (feedforward,backpropagation,...) for each batch
            for i,(img,gt_dmap) in enumerate(train_dataloader):
                img=img.to(device)
                gt_dmap=gt_dmap.to(device)
                    # forward propagation
                est_dmap=self(img)
                #print('img',img.shape,' gt',gt_dmap.shape,'est',est_dmap.shape)
                print('img',img.size(),' gt',gt_dmap.size(),'est',est_dmap.size())
                    # calculate loss
                loss=train_params.criterion(est_dmap,gt_dmap)
                epoch_loss+=loss.item()
                    # Setting gradient to zero ,(only in pytorch , because of backward() that accumulate gradients)
                self.optimizer.zero_grad()
                    # Backpropagation
                loss.backward()
                self.optimizer.step()
                del img,gt_dmap,est_dmap
            print("\t epoch:"+str(epoch)+"\n","\t loss:",epoch_loss/len(train_dataloader))
            
                # Log results in checkpoints2 directory
            epochs_list.append(epoch)
            train_loss_list.append(epoch_loss/len(train_dataloader))
         
                # Set the Model on validation mode
            self.eval()
            MAE=0
            MSE=0
            for i,(img,gt_dmap) in enumerate(test_dataloader):
                img=img.to(device)
                gt_dmap=gt_dmap.to(device)
                    # forward propagation
                est_dmap=self(img)
                MAE+=abs(est_dmap.data.sum()-gt_dmap.data.sum()).item()
                MSE+=np.math.pow(est_dmap.data.sum()-gt_dmap.data.sum(),2)
                del img,gt_dmap,est_dmap
            MAE=MAE/len(test_dataloader)  
            MSE=np.math.sqrt(MSE/len(test_dataloader))

            if MAE<self.min_MAE:
                self.min_MAE=MAE
                self.min_epoch=epoch
            test_error_list.append(MAE)
            print("\t error:"+str(MAE)+" min_MAE:"+str(self.min_MAE)+" min_epoch:"+str(self.min_epoch))
            check_point={
                'model_state_dict':self.state_dict(),
                'optimizer_state_dict':self.optimizer.state_dict(),
                'loss': epoch_loss/len(train_dataloader),
                'mae': MAE,
                'min_MAE': self.min_MAE,
                'min_epoch': self.min_epoch
            }
            self.save_checkpoint(check_point,os.path.join(utils.BASE_PATH,'checkpoints2',self.__class__.__name__,'epoch_'+str(epoch)+'.pth'))
            
        end=time.time()
        self.make_summary(finished=True)
        return (epochs_list,train_loss_list,test_error_list,self.min_epoch,self.min_MAE,str(datetime.timedelta(seconds=end-start)))      
                