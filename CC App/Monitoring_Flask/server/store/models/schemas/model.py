import torch
import torch.nn as NN
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import os,sys
import numpy as np

class Model(NN.Module):

    def __init__(self):
        super(Model, self).__init__()
       
    def build(self, weightsFlag):
        """
            Build Net Architecture
        """
        pass

    def train_model(self, train_dataloader, test_dataloader, train_params, resume=False,new_train=False):
        """
            Start training the model with specified parameters.
        """
       
    def retrain_model(self, params=None):
        pass

    def eval_model(self, test_dataloader, eval_metrics='all'):
        """
            Evaluate/Test the model after train is completed and output performence metrics used for test purpose.
        """
        

    def save_checkpoint(self, chkpt, path):
        """
            Save a checkpoint in the specified path.
        """
        

    
    def load_chekpoint(self, path):
        """
            Load a checkpoint from the specified path in order to resume training.
        """
        
    @staticmethod
    def migrate(optimizer,device):
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        return optimizer            

    @staticmethod
    def save(model):
        """
            Save the whole model. This method is called once training is finished in order to keep the best model.

        """
          

    def make_summary(self, finished=False, test_mse=None, test_mae=None):
        pass
        

