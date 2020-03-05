import os, sys,inspect
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
  
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
    # User's modules from another directory
sys.path.append(parentdir + "\\bases")    
import datasets
from datasets import *


class Loader:
    def load(self,train_size=80,test_size=20,shuffle_flag=True,batch_size=10):pass

    @staticmethod
    def merge_datasets(loaders_list:list,shuffleFlag=True):
        train_set=[]
        test_set=[]
        dataset=torch.utils.data.Dataset()
        for train_loader,test_loader in loaders_list:
            for index,features in enumerate(train_loader):
                train_set.append(features)
            for index,features in enumerate(test_loader):
                test_set.append(features)  

        if shuffleFlag:
            np.random.shuffle(train_set)
            np.random.shuffle(test_set)

        train_dataset=BasicDataSet(train_set)    
        test_dataset=BasicDataSet(test_set)   
        return train_dataset,test_dataset
                  
class SimpleLoader(Loader):

    def __init__(self,img_rootPath,gt_dmap_rootPath):
        self.img_rootPath=img_rootPath
        self.gt_dmap_rootPath=gt_dmap_rootPath


    def load(self,train_size=80,test_size=20,shuffle_flag=True,batch_size=10):
        dataset=CrowdDataset(self.img_rootPath,self.gt_dmap_rootPath) 
        self.dataSet_size=len(dataset)
        indices = list(range(self.dataSet_size))
        split = int(np.floor(test_size * self.dataSet_size))

        random_seed=30
        if shuffle_flag:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_sampler = SubsetRandomSampler(list(indices[split:]))
        test_sampler = SubsetRandomSampler(list(indices[:split]))    

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=test_sampler)

        return [(train_loader,test_loader)]

class GenericLoader(Loader):

    def __init__(self,img_gt_dmap_list):
        self.img_gt_dmap_list=img_gt_dmap_list
        


    def load(self,train_size=80,test_size=20,shuffle_flag=True,batch_size=10):
        all_datasets=[]
        for img_root_path,dm_root_path in self.img_gt_dmap_list:
            dataset=CrowdDataset(img_root_path,dm_root_path) 
            self.dataSet_size=len(dataset)
            indices = list(range(self.dataSet_size))
            split = int(np.floor(test_size * self.dataSet_size))

            random_seed=30
            if shuffle_flag:
                np.random.seed(random_seed)
                np.random.shuffle(indices)

            train_sampler = SubsetRandomSampler(list(indices[split:]))
            test_sampler = SubsetRandomSampler(list(indices[:split]))    

            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
            test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=test_sampler)
            
            all_datasets.append(train_loader,test_loader)                                        

        return all_datasets                                                