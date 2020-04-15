import torch
import torch.nn as NN
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import os,sys,inspect,glob,random,re,time,datetime
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
    # User's module from another directory
sys.path.append(os.path.join(parentdir, "bases"))   
import utils 
import storagemanager
from params import *
from torch import Tensor

class Model(NN.Module):

    def __init__(self):
        super(Model,self).__init__()
        self.params=TrainParams.defaultTrainParams()
        
    def build(self,weightsFlag):
        '''
            Build Net Architecture
        '''
        pass    
        

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
                # self.load_state_dict(torch.load(last_epoch))
                
                # last_model=torch.load(last_epoch.replace('.pth','.pkl'))
                # #self.optimizer=last_model.optimizer
                # if hasattr(last_model,'min_MAE'):self.min_MAE=last_model.min_MAE
                # if hasattr(last_model,'min_epoch'):self.min_epoch=last_model.min_epoch
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
            
            # torch.save(self.state_dict(),os.path.join(utils.BASE_PATH,'checkpoints2/epoch_'+str(epoch)+'.pth'))
            # torch.save(self,os.path.join(utils.BASE_PATH,'checkpoints2/epoch_'+str(epoch)+".pkl"))

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
            # vis.line(win=1,X=epochs_list, Y=train_loss_list, opts=dict(title='train_loss'))
            # vis.line(win=2,X=epochs_list, Y=test_error_list, opts=dict(title='test_error'))
            # show an image
            # index=random.randint(0,len(test_dataloader)-1)
            # img,gt_dmap=test_dataset[index]
            # vis.image(win=3,img=img,opts=dict(title='img'))
            # vis.image(win=4,img=gt_dmap/(gt_dmap.max())*255,opts=dict(title='gt_dmap('+str(gt_dmap.sum())+')'))
            # img=img.unsqueeze(0).to(device)
            # gt_dmap=gt_dmap.unsqueeze(0)
            # est_dmap=self(img)
            # est_dmap=est_dmap.squeeze(0).detach().cpu().numpy()
            # vis.image(win=5,img=est_dmap/(est_dmap.max())*255,opts=dict(title='est_dmap('+str(est_dmap.sum())+')'))
        end=time.time()
        self.make_summary(finished=True)
        return (epochs_list,train_loss_list,test_error_list,self.min_epoch,self.min_MAE,str(datetime.timedelta(seconds=end-start)))

    def retrain_model(self,params=None):
        pass

    def eval_model(self,test_dataloader,eval_metrics='all'):
        '''
            Evaluate/Test the model after train is completed and output performence metrics used for test purpose.
        '''
        print("####### Validating The model...")
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.eval()
        MAE=0
        MSE=0
        cpt=0
        with torch.no_grad():
            for i,(img,gt_dmap) in enumerate(test_dataloader):
                    # Transfer input and target to Device(GPU/CPU)
                img=img.to(device)
                gt_dmap=gt_dmap.to(device)

                    # Forward propagation
                est_dmap=self(img.squeeze(0))
                
                MAE+=abs(est_dmap.data.sum()-gt_dmap.data.sum()).item()
                MSE+=np.math.pow(est_dmap.data.sum()-gt_dmap.data.sum(),2)

                    # Show the estimated density map via matplotlib
                if cpt%10==0: 
                    est_dmap=est_dmap.squeeze(0).squeeze(0).cpu().numpy()
                    plt.imshow(est_dmap,cmap=CM.jet)
                del img,gt_dmap,est_dmap
            MAE=MAE/len(test_dataloader)  
            MSE=np.math.sqrt(MSE/len(test_dataloader))
        print("\t Test MAE : ",MAE,"\t test MSE : ",MSE)    
        return (MAE,MSE)         
                

    def save_checkpoint(self,chkpt,path):
        '''
            Save a checkpoint in the specified path.
        '''
            # If the directory doesn't exist, create it.
        utils.make_path(os.path.split(path)[0])
        # torch.save(chkpt, path) 
        if 'drive/My Drive' in path:
            print('google colab')
        env='drive' if 'drive/My Drive' in path else 'os'
        storagemanager.save_file(path,chkpt,env)

    def load_chekpoint(self,path):
        '''
            Load a checkpoint from the specified path in order to resume training.
        '''
        chkpt=torch.load(path)
        self.load_state_dict(chkpt['model_state_dict'])
        self.optimizer.load_state_dict(chkpt['optimizer_state_dict'])

        return chkpt['loss'],chkpt['min_MAE'],chkpt['min_epoch']

    def save(self):
        '''
            Save the whole model. This method is called once training is finished in order to keep the best model.

        '''
        path=os.path.join(utils.BASE_PATH,'obj','models',self.__class__.__name__+'.pkl')
        utils.make_path(os.path.split(path)[0])
        torch.save(self,path) 

    def make_summary(self,finished=False):
        path=os.path.join(utils.BASE_PATH,'checkpoints2',self.__class__.__name__,'summary.json')
        summary={
            'status': finished,
            'min_epoch':self.min_epoch,
            'min_loss':0,
            'min_MAE':self.min_MAE,
            'train_params':{
                'lr':self.params.lr,
                'momentum':self.params.momentum,
                'maxEpochs':self.params.maxEpochs,
                'criterionMethode':self.params.criterion.__class__.__name__,
                'optimizationMethod':self.params.optimizer.__class__.__name__
            }
            
        }
        utils.make_path(os.path.split(path)[0])
        utils.save_json(summary,path)



    
        