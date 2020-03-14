import os, sys,inspect
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
  
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
    # User's modules from another directory
sys.path.append(os.path.join(parentdir, "bases"))     
import datasets
import utils
from datasets import *
from utils import *


class Loader:
    def __init__(self,reset_samplers=True):
        self.reset_samplers_flag=reset_samplers

    def load(self,train_size=80,test_size=20,shuffle_flag=True,batch_size=1):pass

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

    def __init__(self,img_rootPath,gt_dmap_rootPath,reset_samplers=True):
        super(SimpleLoader,self).__init__(reset_samplers)
        self.img_rootPath=img_rootPath
        self.gt_dmap_rootPath=gt_dmap_rootPath


    def load(self,train_size=80,test_size=20,shuffle_flag=True,batch_size=1):
        dataset=CrowdDataset(self.img_rootPath,self.gt_dmap_rootPath) 
        self.dataSet_size=len(dataset)
        indices = list(range(self.dataSet_size))
        split = int(np.floor(test_size * self.dataSet_size/100))

        load_flag=False
        if not self.reset_samplers_flag and utils.path_exists('./obj/loaders/samplers.pkl'):
            print("\t Found a sampler restore point...")  
            samplers_recov=torch.load('../../obj/loaders/samplers.pkl')  
            for obj in samplers_recov:
                if obj['self.img_rootPath']==self.img_rootPath and obj['self.gt_dmap_rootPath']==self.gt_dmap_rootPath:
                    train_sampler=obj['train_sampler']
                    test_sampler=obj['test_sampler']
                    load_flag=True
            

        if not load_flag:
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

    def __init__(self,img_gt_dmap_list,reset_samplers=False):
        super(GenericLoader,self).__init__(reset_samplers)
        self.img_gt_dmap_list=img_gt_dmap_list
        
        


    def load(self,train_size=80,test_size=20,shuffle_flag=True,batch_size=1):
        all_datasets=[]
        load_flag=False
        if not self.reset_samplers_flag and utils.path_exists('./obj/loaders/samplers.pkl'):
            print("\t Found a sampler restore point...")  
            samplers_recov=torch.load('../../obj/loaders/samplers.pkl')  
            img_paths=[obj['train_sampler'] for obj in self.img_gt_dmap_list]
            gt_map_paths=[obj['test_sampler'] for obj in self.img_gt_dmap_list]
            load_flag=True
            
            list_samplers=[obj for obj in samplers_recov
                            if obj['img_rootPath']in img_paths and obj['dm_root_path'] in gt_map_paths]
                
                # if ==self.img_rootPath and ==self.gt_dmap_rootPath:
                #     train_sampler=obj['train_sampler']
                #     test_sampler=obj['test_sampler']
                #     load_flag=True
        if load_flag:
            print("\t Found dataset from restor point, loading samples....")
            for obj in list_samplers:
                dataset=CrowdDataset(obj['img_rootPath'],obj['dm_root_path'])
                all_datasets.append((torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                                sampler=obj['train_sampler']),

                                    torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                        sampler=obj['test_sampler'])
                                                )
                                    )
            print("\t Done. Dataset restored.")

        else:
            for img_root_path,dm_root_path in self.img_gt_dmap_list:
                dataset=CrowdDataset(img_root_path,dm_root_path) 
                dataSet_size=len(dataset)
                
                indices = list(range(dataSet_size))
                split = int(np.floor(test_size * dataSet_size/100))

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
                
                all_datasets.append((train_loader,test_loader))                                        

        return all_datasets                                                