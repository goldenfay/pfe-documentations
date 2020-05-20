import torch
import numpy as np
import os,sys,inspect,glob,traceback
import importlib

# currentdir = os.path.dirname(os.path.abspath(
#     inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)    
# sys.path.append(os.path.join(parentdir,'ML_package','models'))

sys.path.append('utils')
sys.path.append('utils/YOLO_V3')
sys.path.append('utils/mobilenet_SSD')
import store.models.equivalence as equivalence
import utils.YOLO_V3.detector as yolo_detector 
import utils.mobilenet_SSD.bbox_counter as ssd_detector
from utils.detection_model import DetectionModel

def is_detection_model(model_name):
    return model_name in ['mobileSSD','yolo']

def is_densitymap_model(model):
    return model.__class__.__name__ in ['SANet','CSRNet','MCNN','CCNN']

class ModelManager:
        
    model=None
    device =torch.device('cpu')
    BASE_PATH=''
    

    @staticmethod
    def get_instance(model_name):
        if model_name=="MCNN":
            return MCNN(True)
        elif model_name=="CSRNet":
            return CSRNet(True)
        elif model_name=="SANet":
            return SANet(True)            
        elif model_name=="CCNN":
            return CCNN(True)
        else: raise AttributeError('Invalid argument : model name')    

    @classmethod
    def load_model(cls,model_name):
        
        return torch.load(os.path.join(cls.FROZEN_MODELS_PATH,model_name+'.pth'),map_location=cls.device)
    @classmethod
    def load_detection_model(cls,model_name):
        # cls.model=torch.load(os.path.join(cls.FROZEN_MODELS_PATH,model_name+'.pth'),map_location=cls.device)
        
        if model_name=='mobileSSD':
            cls.model=ssd_detector.MobileSSD(model_name)
        else:
            cls.model=yolo_detector.YOLO(model_name)    
        return cls.model

    @classmethod
    def process_frame(cls,frame):
            #If the current model is a density map based model
        if is_densitymap_model(cls.model):
            if not isinstance(frame,torch.Tensor):
                frame=torch.Tensor(frame).permute((2,0,1))
            cls.model.to(cls.device)   
            frame.to(cls.device)
            if len(frame.shape)==3:
                frame=frame.unsqueeze(0) 
            dmap=cls.model(frame)

            return dmap.squeeze().detach().cpu().numpy(),dmap.data.sum()
        else: # It's a detection model
            return cls.model.forward(frame)

           


    @staticmethod
    def load_external_model(model_name):
        try:
            match_dict=getattr(equivalence,model_name+'_DICT_MATCH')
            pretrained_dict = ModelManager.load_model(model_name)
            
            model=ModelManager.get_instance(model_name)
            model_dict = model.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {match_dict[k]: v for k, v in pretrained_dict.items() if k in match_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            model.load_state_dict(model_dict)
            ModelManager.model=model
            return model
        except Exception as e:
            print('Couldn''t load equivalence dict. Model type is invalide',end='\n\t')
            traceback.print_exc()    
    @classmethod
    def set_base_path(cls,path):
        cls.BASE_PATH=path
        cls.FROZEN_MODELS_PATH=os.path.join(cls.BASE_PATH,'frozen')
        cls.MODELS_SCHEMA_PATH=os.path.join(cls.BASE_PATH,'schemas')
        sys.path.append(cls.BASE_PATH)
        sys.path.append(cls.MODELS_SCHEMA_PATH)
        # importlib.import_module(cls.MODELS_SCHEMA_PATH)
        global Model,CCNN,CSRNet,MCNN,SANet 
        import schemas
        from model import Model
        from CCNN import CCNN
        from CSRNet import CSRNet
        from mcnn import MCNN
        from SANet import SANet
        # print(schemas)

