import torch
import numpy as np
from matplotlib import cm
import os,sys,glob,traceback,time
import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import FPS

sys.path.append('utils')
sys.path.append('utils/YOLO_V3')
sys.path.append('utils/mobilenet_SSD')
    # User's modules
import utils.YOLO_V3.yolo_model as yolo_detector
import utils.mobilenet_SSD.bbox_counter as ssd_detector
from utils.detection_model import DetectionModel
import store.models.equivalence as equivalence

# QUEUE=multiprocessing.Queue()

def is_detection_model(model_name):
    return model_name in ['mobileSSD', 'yolo']


def is_densitymap_model(model):
    return model.__class__.__name__ in ['SANet', 'CSRNet', 'MCNN', 'CCNN']


class ModelManager:

    model = None
    device = torch.device('cpu')
    BASE_PATH = ''

    @staticmethod
    def get_instance(model_name):
        if model_name == "MCNN":
            return MCNN(True)
        elif model_name == "CSRNet":
            return CSRNet(True)
        elif model_name == "SANet":
            return SANet(True)
        elif model_name == "CCNN":
            return CCNN(True)
        else:
            raise AttributeError('Invalid argument : model name')

    @classmethod
    def load_model(cls, model_name):

        return torch.load(os.path.join(cls.FROZEN_MODELS_PATH, model_name+'.pth'), map_location=cls.device)

    @classmethod
    def load_detection_model(cls, model_name):
        # cls.model=torch.load(os.path.join(cls.FROZEN_MODELS_PATH,model_name+'.pth'),map_location=cls.device)

        if model_name == 'mobileSSD':
            cls.model = ssd_detector.MobileSSD()
        else:
            cls.model = yolo_detector.YOLO()
        return cls.model

    @classmethod
    def process_frame(cls, frame):
        #TO REMEMBER : padding = int((kernel_size - 1) / 2) if same_padding else 0
            # If the current model is a density map based model
        if is_densitymap_model(cls.model):
            if not isinstance(frame, torch.Tensor):
                frame = torch.Tensor(frame).permute((2, 0, 1))
            with torch.no_grad():
                cls.model.to(cls.device)
                frame.to(cls.device)
                if len(frame.shape) == 3:
                    frame = frame.unsqueeze(0)
                dmap = cls.model(frame)
                dmap=dmap+(abs(dmap.min().item()))
                count = dmap.data.sum().item()
                if cls.model.__class__.__name__ == 'SANet':
                    count = count//100
                elif cls.model.__class__.__name__ == 'CSRNet':
                    count = count//80
            return dmap.squeeze().detach().cpu().numpy(), count
        else:  # It's a detection model
            return cls.model.forward(frame)

    @classmethod
    def process_video(cls, video_path, args=None,queue=None):
        output=args.get('output',None)
        
            # If the current model is a density map based model
        if is_densitymap_model(cls.model):
            vs = cv2.VideoCapture(video_path)
            frame = vs.read()
                # VideoStream returns a frame, VideoCapture returns a tuple
            frame = frame[1] if len(frame) > 1 else frame

            if frame is None:
                raise Exception('[FATAL] Cannot read video stream')
                traceback.print_exc()

            (H, W) = frame.shape[:2]
            
            writer=None
            fps = FPS().start()
            totalFrame=0
          
            while True:
                frame = imutils.resize(frame, height=500)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                (H, W) = frame.shape[:2]
                if totalFrame%30==0:
                    start=time.time()
                    dmap,count=ModelManager.process_frame(rgb_frame)
                    elapsed=time.time()-start
                    text = "Estimated count : {}".format(count)
                    cv2.putText(frame, text, (10, H - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # print('Processed in : ',elapsed,' ',text)            
                    cv2.imshow("Frame", frame)
                    # cv2.imshow("DM", dmap)
                    if output is not None and writer is None:
                        min_height=min(dmap.shape[0],frame.shape[0])
                        frame=imutils.resize(frame, height=min_height)
                        dmap = imutils.resize(dmap, height=min_height)
                        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                        writer = cv2.VideoWriter(os.path.join(output, 'output', 'CNNCC_output.avi'), fourcc, 30,(dmap.shape[1]+frame.shape[1],min_height), True)
                    
                    if writer is not None:
                        dmap = imutils.resize(dmap, height=min_height)
                        frame=imutils.resize(frame, height=min_height)
                        dmap=cm.jet(dmap)*255
                        dmap=dmap[:,:,:3].astype('uint8')
                        dmap=cv2.cvtColor(dmap, cv2.COLOR_BGR2RGB)
                        # concated = cv2.vconcat(frame, dmap)
                        nb_rows= int(dmap.shape[0]+frame.shape[0])
                        nb_cols= int(dmap.shape[1]+frame.shape[1])
                        concated = np.zeros(shape=(frame.shape[0], nb_cols, 3), dtype=np.uint8)
                        concated[:,:frame.shape[1]]=frame
                        concated[:,frame.shape[1]:]=dmap[:,:,:3]
                        print(frame.shape,dmap.shape,concated.shape)
                        cv2.imshow("DM", concated)
                        writer.write(concated)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                fps.update()
                frame = vs.read()
                    # VideoStream returns a frame, VideoCapture returns a tuple
                frame = frame[1] if len(frame) > 1 else frame

                if frame is None:
                    break
            fps.stop()
            print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
            print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
            vs.release()

            if writer is not None:
                writer.release()
        else:  # It's a detection model
            if args is None:
                args = {
                    'input': video_path,
                    'silent': True,
                    'write_output': True,

                }
            # cls.model.forward_video(args)
            for x in cls.model.forward_video(args):
                # queue.put_nowait(x)
                yield x
  

    @staticmethod
    def load_external_model(model_name,external=True):
        try:
            model = ModelManager.get_instance(model_name)
            if external:
                pretrained_dict, match_dict = equivalence.get_dict_match(
                    model_name)
                model_dict = model.state_dict()

                # 1. filter out unnecessary keys
                pretrained_dict = {
                    match_dict[k]: v for k, v in pretrained_dict.items() if k in match_dict}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                model.load_state_dict(model_dict)
                
            else:
                model.load_state_dict(torch.load(os.path.join(ModelManager.FROZEN_MODELS_PATH,'internal',model_name+'.pth'),map_location=ModelManager.device)['model_state_dict'])    
            ModelManager.model = model
            return model
        except Exception as e:
            print('Couldn''t load equivalence dict. Model type is invalide', end='\n\t')
            traceback.print_exc()

    @classmethod
    def set_base_path(cls, path):
        cls.BASE_PATH = path
        cls.FROZEN_MODELS_PATH = os.path.join(cls.BASE_PATH, 'frozen')
        cls.MODELS_SCHEMA_PATH = os.path.join(cls.BASE_PATH, 'schemas')
        sys.path.append(cls.BASE_PATH)
        sys.path.append(cls.MODELS_SCHEMA_PATH)

        global Model, CCNN, CSRNet, MCNN, SANet
        import schemas
        from model import Model
        from CCNN import CCNN
        from CSRNet import CSRNet
        from mcnn import MCNN
        from SANet import SANet



import os,sys
import cv2,imutils
ROOT_PATH='C:\\Users\\PC\\Desktop\\PFE related\\applications\\CC App\\Developpement'
VIDEO_PATH='C:\\Users\\PC\\Downloads\\Video\\hadj.mp4'
model_type='SANet'
sys.path.append(ROOT_PATH)
import config
from modelmanager import ModelManager
ModelManager.set_base_path(config.FROZEN_MODELS_BASE_PATH)
ModelManager.load_external_model(model_type)
for x in ModelManager.process_video(VIDEO_PATH,args=dict(output=os.path.join(ROOT_PATH,'ressources','videos'))):
    # cv2.imshow('frame22222',x)
    pass
