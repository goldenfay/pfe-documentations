import torch
import torch.nn as NN
import torch.nn.functional as F
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
from gitmanager import *
from params import *
from torch import Tensor

class Model(NN.Module):

    def __init__(self):
        super(Model,self).__init__()
        self.params=TrainParams.defaultTrainParams()
        self.checkpoints_dir=os.path.join(utils.BASE_PATH,'checkpoints2',self.__class__.__name__)
        
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
            params_hist=[int(re.sub("[^0-9]+","",file_path[list(re.finditer("[\\\/]",file_path))[-1].start(0):])) for file_path in glob.glob(os.path.join(os.path.join(self.checkpoints_dir),'*.pth'))]
            
            

            if len(params_hist)>0:
                print("\t Restore Checkpoints2 found! Resuming training...")
                sorted_hist=sorted(params_hist)
                start_epoch=max(sorted_hist)
                last_epoch=glob.glob(os.path.join(os.path.join(self.checkpoints_dir,'epoch_'+str(start_epoch)+'.pth')))[0]
                
                # #self.optimizer=last_model.optimizer
                # if hasattr(last_model,'min_MAE'):self.min_MAE=last_model.min_MAE
                # if hasattr(last_model,'min_epoch'):self.min_epoch=last_model.min_epoch
                    #//////////
                _,self.min_MAE,self.min_epoch=self.load_chekpoint(last_epoch)

                files_to_push=[]
                for epoch in sorted_hist:
                    if epoch!=self.min_epoch and epoch!=start_epoch:
                        path= glob.glob(os.path.join(os.path.join(self.checkpoints_dir,'epoch_'+str(epoch)+'.pth')))[0]
                        obj=torch.load(path)
                        if obj['model_state_dict'] is not None or obj['optimizer_state_dict']is not None:
                            obj['model_state_dict']=None 
                            obj['optimizer_state_dict']=None
                            self.save_checkpoint(obj,path)
                            files_to_push.append(path)
                
                git_manager=GitManager(user='ihasel2020@gmail.com',pwd='pfemaster2020')  
                git_manager.authentification()
                target_repo=git_manager.get_repo('checkpoints') 
                res=git_manager.push_files(target_repo,files_to_push,'checkpoints migration')
                if isinstance(res,int)and res==len(files_to_push):
                    print('\t Successfully comitted previous checkpoints(',res,' files).')     

                else :  raise RuntimeError('Couldn\'t push all files')
  
                            

                        

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
                if not est_dmap.size()==gt_dmap.size():
                    
                    est_dmap=F.interpolate(est_dmap,size=(gt_dmap.size()[2],gt_dmap.size()[3]),mode='bilinear')
                
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
                if not est_dmap.size()==gt_dmap.size():
                    
                    est_dmap=F.interpolate(est_dmap,size=(gt_dmap.size()[2],gt_dmap.size()[3]),mode='bilinear')

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
            self.save_checkpoint(check_point,os.path.join(self.checkpoints_dir,'epoch_'+str(epoch)+'.pth'))
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
        env='drive' if 'drive/My Drive' in path else 'os'
        flag=storagemanager.save_file(path,chkpt,env,self.min_epoch)

        if flag==0: # There isn't available space on drive
            print("\t Optimizing space...")
            parent_path=os.path.split(path)[0]
            sorted_hist= sorted([int(re.sub("[^0-9]+","",file_path[list(re.finditer("[\\\/]",file_path))[-1].start(0):])) for file_path in glob.glob(os.path.join(parent_path,'*.pth'))])
            files_to_push=[]
            for epoch in sorted_hist:
                    if epoch!=self.min_epoch:
                        path= glob.glob(os.path.join(os.path.join(parent_path,'epoch_'+str(epoch)+'.pth')))[0]
                        obj=torch.load(path)
                        if obj['model_state_dict'] is not None or obj['optimizer_state_dict']is not None:
                            obj['model_state_dict']=None 
                            obj['optimizer_state_dict']=None
                            self.save_checkpoint(obj,path)
                            files_to_push.append(path)
            print("\t Pushing checkpoints to github...")    
            git_manager=GitManager(user='ihasel2020@gmail.com',pwd='pfemaster2020')  
            git_manager.authentification()
            target_repo=git_manager.get_repo('checkpoints') 
            res=git_manager.push_files(target_repo,files_to_push,'checkpoints migration')
            if isinstance(res,int)and res==len(files_to_push):
                print('\t Successfully comitted previous checkpoints(',res,' files).')     

            else :  raise RuntimeError('Couldn\'t push all files')

            torch.save(chkpt, path) 


    def load_chekpoint(self,path):
        '''
            Load a checkpoint from the specified path in order to resume training.
        '''
        chkpt=torch.load(path).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
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
        path=os.path.join(self.checkpoints_dir,'summary.json')
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




    
        
