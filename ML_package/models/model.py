import torch
import torch.nn as NN
import pickle
import matplotlib as plt
import matplotlib.cm as CM
import os, sys,inspect
import numpy as npmath

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
    # User's module from another directory
sys.path.append(os.path.join(parentdir, "bases"))   

from params import *
class Model(NN.Module):

    def __init__(self):
        super(Model,self).__init__()
        self.params=TrainParams.defaultTrainParams()
        
    def build(self):
        pass    
        

    def train_model(self,train_dataloader,test_dataloader,train_params:TrainParams,resume=False):
        pass

    def retrain_model(self,params=None):
        pass

    def eval_model(self,test_dataloader,eval_metrics='all'):
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
                est_dmap=self(img).detach()
                
                MAE+=abs(est_dmap.data.sum()-gt_dmap.data.sum()).item()
                MSE+=numpy.math.pow(est_dmap.data.sum()-gt_dmap.data.sum(),2)

                    # Show the estimated density map via matplotlib
                if cpt%10==0: 
                    est_dmap=est_dmap.squeeze(0).squeeze(0).cpu().numpy()
                    plt.imshow(est_dmap,cmap=CM.jet)
                del img,gt_dmap,est_dmap
            MAE=MAE/len(test_dataloader)  
            MSE=numpy.math.sqrt(MSE/len(test_dataloader))
        print("\t Test MAE : ",MAE,"\t test MSE : ",MSE)    
        return (MAE,MSE)         
                
    def save(self):
        pass

    
        