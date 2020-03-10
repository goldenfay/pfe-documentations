import torch
from torch import nn
import visdom as vis
import numpy as np
import os,sys,glob,random,re
    # User's modules
import model
from model import *



class MCNN(Model):
    
    def __init__(self,weightsFlag=False):
        super(MCNN,self).__init__()
        self.build(weightsFlag)


    def build(self,weightsFlag):
        print("####### Building Net architecture...")
        
        self.branch1=nn.Sequential(
            nn.Conv2d(3,16,9,padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,7,padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32,16,7,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,8,7,padding=3),
            nn.ReLU(inplace=True)
        )

        self.branch2=nn.Sequential(
            nn.Conv2d(3,20,7,padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20,40,5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(40,20,5,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(20,10,5,padding=2),
            nn.ReLU(inplace=True)
        )

        self.branch3=nn.Sequential(
            nn.Conv2d(3,24,5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48,24,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24,12,3,padding=1),
            nn.ReLU(inplace=True)
        )

        self.fuse=nn.Sequential(nn.Conv2d(30,1,1,padding=0))

        if not weightsFlag:
            self._initialize_weights()
              

    def forward(self,img_tensor):
        if len(img_tensor.shape)==3: 
            
            img_tensor=torch.tensor(img_tensor[np.newaxis,:,:,:],dtype=torch.float)
              
        branch1=self.branch1(img_tensor)
        branch2=self.branch2(img_tensor)
        branch3=self.branch3(img_tensor)
        x=torch.cat((branch1,branch2,branch3),1)
        x=self.fuse(x)
        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0) 

    def train_model(self,train_dataloader,test_dataloader,train_params:TrainParams,resume=False):
        print("####### Training The model...")
        self.optimizer=train_params.optimizer
            # Get the device (GPU/CPU) and migrate the model to it
        device=train_params.device
        print("\t Setting up model on ",device.type,"...")    
        self.to(device)
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
           
        
            # Initialize training variables
        print("\t Initializing ","...")    
        min_MAE=10000
        min_epoch=0
        epochs_list=[]
        train_loss_list=[]
        test_error_list=[]
        start_epoch=1
         # If resume option is specified, restore state of model and resume training
            
        if resume:
            params_hist=glob.glob(os.path.join('checkpoints','*.param'))
            if len(params_hist)>0:
                print("\t Restore Checkpoints found! Resuming training...")
                self.load_state_dict(torch.load(params_hist[-1]))
                start_epoch=int(re.sub("[^0-9]+","",params_hist[-1][list(re.finditer("[\\\/]",params_hist[-1]))[-1].start(0):]))
                print(self.state_dict())
            # Start Train
        for epoch in range(start_epoch,train_params.maxEpochs):
                # Set the Model on training mode
            self.train()
            epoch_loss=0
                # Run training pass (feedforward,backpropagation,...)
            for i,(img,gt_dmap) in enumerate(train_dataloader):
                img=img.to(device)
                gt_dmap=gt_dmap.to(device)
                    # forward propagation
                et_dmap=self(img)
                    # calculate loss
                loss=train_params.criterion(et_dmap,gt_dmap)
                epoch_loss+=loss.item()
                    # Setting gradient to zero ,(only in pytorch , because of backward() that accumulate gradients)
                self.optimizer.zero_grad()
                    # Backpropagation
                loss.backward()
                self.optimizer.step()
            #print("epoch:",epoch,"loss:",epoch_loss/len(dataloader))
                # Log results in checkpoints directory
            epochs_list.append(epoch)
            train_loss_list.append(epoch_loss/len(train_dataloader))
            torch.save(self.state_dict(),'./checkpoints/epoch_'+str(epoch)+".param")
            torch.save(self,'./checkpoints/epoch_'+str(epoch)+".pkl")
                # Set the Model on validation mode
            self.eval()
            MAE=0
            for i,(img,gt_dmap) in enumerate(test_dataloader):
                img=img.to(device)
                gt_dmap=gt_dmap.to(device)
                    # forward propagation
                et_dmap=self(img)
                MAE+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
                del img,gt_dmap,et_dmap
            if MAE/len(test_dataloader)<min_MAE:
                min_MAE=MAE/len(test_dataloader)
                min_epoch=epoch
            test_error_list.append(MAE/len(test_dataloader))
            print("\t epoch:"+str(epoch)+"\n\t error:"+str(MAE/len(test_dataloader))+" min_MAE:"+str(min_MAE)+" min_epoch:"+str(min_epoch))
            # vis.line(win=1,X=epochs_list, Y=train_loss_list, opts=dict(title='train_loss'))
            # vis.line(win=2,X=epochs_list, Y=test_error_list, opts=dict(title='test_error'))
            # show an image
            # index=random.randint(0,len(test_dataloader)-1)
            # img,gt_dmap=test_dataset[index]
            # vis.image(win=3,img=img,opts=dict(title='img'))
            # vis.image(win=4,img=gt_dmap/(gt_dmap.max())*255,opts=dict(title='gt_dmap('+str(gt_dmap.sum())+')'))
            # img=img.unsqueeze(0).to(device)
            # gt_dmap=gt_dmap.unsqueeze(0)
            # et_dmap=self(img)
            # et_dmap=et_dmap.squeeze(0).detach().cpu().numpy()
            # vis.image(win=5,img=et_dmap/(et_dmap.max())*255,opts=dict(title='et_dmap('+str(et_dmap.sum())+')'))
                             

if __name__=="__main__":
    import matplotlib.pyplot as plt
    # img=torch.rand((3,800,1200),dtype=torch.float)
    img_rootPath="C:\\Users\\PC\\Desktop\\PFE related\\existing works\\Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper code sample\\MCNN-pytorch-master\\MCNN-pytorch-master\\ShanghaiTech\\part_A\\train_data\\images"

    img=plt.imread(os.path.join(img_rootPath,"IMG_10.jpg"))
    img=torch.from_numpy(img.transpose(2,0,1))
    mcnn=MCNN()
    out_dmap=mcnn(img)
    # print(out_dmap.shape)
    # plt.imshow(  img  )
    # plt.imshow(  out_dmap.permute()  )
    # plt.show()
    x=vis.Visdom()
    # x.images(img,1,10)
    x.image(win=5,img=img,opts=dict(title='img'))
    x.image(win=5,img=out_dmap/(out_dmap.max())*255,opts=dict(title='et_dmap('))
    