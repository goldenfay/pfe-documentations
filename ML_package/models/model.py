import torch
import torch.nn as NN
import pickle
import os, sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
    # User's module from another directory
sys.path.append(parentdir + "\\bases")   

from params import *
class Model(NN.Module):

    def __init__(self):
        super(Model,self).__init__()
        self.params=TrainParams.defaultTrainParams()
        
        
        

    def train_model(self,train_dataloader,test_dataloader,train_params:TrainParams):
        pass

    def retrain_model(self,params=None):
        pass

    def eval_model(self):
        pass
    def save(self):
        pass

    def build(self):
        pass
        