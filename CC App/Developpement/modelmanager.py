import torch
import numpy as np
from matplotlib import cm
import os,sys,glob,inspect,traceback,time
import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import queue

sys.path.append('utils')
sys.path.append('utils/YOLO_V3')
sys.path.append('utils/mobilenet_SSD')
    # User's modules
import utils.YOLO_V3.yolo_model as yolo_detector
import utils.mobilenet_SSD.bbox_counter as ssd_detector
from utils.detection_model import DetectionModel
import store.models.equivalence as equivalence

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
QUEUE=queue.Queue()

def is_detection_model(model_name):
    return model_name in ['mobileSSD', 'yolo']


def is_densitymap_model(model):
    return model.__class__.__name__ in ['SANet', 'CSRNet', 'MCNN', 'CCNN']


class ModelManager:

    model = None
    outputs_path=os.path.join(currentdir,'ressources','videos','output')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
                cls.model=cls.model.to(cls.device)
                frame=frame.to(cls.device)
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
        output=args.get('output',None) if args is not None else None
        if output is None:
            output=os.path.join(currentdir,'ressources','videos','output')
        if not os.path.exists(output):
            os.makedirs(output)

            # If the current model is a density map based model
        if is_densitymap_model(cls.model):
            print('\t Passing :',args)
            show_regions=False
            tang,b=None,None
            if args is not None and args.get('regions_params',False):
                show_regions=args['regions_params'].get('show',False)
                tang,b=args['regions_params'].get('tang',None),args['regions_params'].get('b',None)
                line_eq=lambda x: int(tang*x+b)
                horizontal_splited=abs(tang)<1

            log_count=args is not None and args.get('log_counts',False)
	
            if log_count:
                log_count_fcn=args.get('log_count_fcn',False)


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
                # frame = imutils.resize(frame, height=500)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                (H, W) = frame.shape[:2]
                if totalFrame%30==0:
                    start=time.time()
                    dmap,count=ModelManager.process_frame(rgb_frame)
                    elapsed=time.time()-start
                    if show_regions:
                        shape=dmap.shape
                        coord_ref=1 if horizontal_splited else 0
                        zoneA=np.zeros(shape)
                        zoneB=np.zeros(shape)
                        # zoneA=np.where(dmap[coord_ref]<line_eq(dmap[coord_ref]),dmap,0)
                        l1=[]
                        l2=[]
                        
                        for row in range(shape[0]):
                            for col in range(shape[1]):
                                couple=(row,col)
                                if couple[coord_ref]<line_eq(couple[coord_ref]):
                                    l1.append(couple)
                                else:
                                    l2.append(couple)

                        for couple in l1:
                            x,y=couple[0],couple[1]
                            zoneA[x,y]=dmap[x,y]
                        for couple in l2:
                            x,y=couple[0],couple[1]
                            zoneB[x,y]=dmap[x,y]
                        countA,countB=int(np.sum(zoneA)),int(np.sum(zoneB))    
                        if cls.model.__class__.__name__ == 'SANet':
                            countA = countA//100
                            countB = countB//100
                        print(countA,countB)
                        # cv2.imshow("Surveillence", zoneA)
                        # cv2.imshow("Surveillence2", zoneB)

                    if show_regions:
                        cv2.line(frame,(0,line_eq(0)),(frame.shape[1],line_eq(frame.shape[1])),(0,200,0),5)	
                        if 'zoneA' in locals() and 'zoneA' in locals():
                            cv2.putText(frame, 'Zone A :  {}'.format(countA), (10,20),
					                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) 
                            cv2.putText(frame, 'Zone B : {}'.format(countB), (frame.shape[1]-50,frame.shape[0]-20),
					                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) 

                    text = "Estimated count : {}".format(count)
                    cv2.putText(frame, text, (10, H - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    print('Processed in : ',elapsed,'s, ',text)            
                  
                if output is not None and writer is None:
                    min_height=min(dmap.shape[0],frame.shape[0])
                    frame=imutils.resize(frame, height=min_height)
                    dmap = imutils.resize(dmap, height=min_height)
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter(os.path.join(output, os.path.basename(video_path).replace('mp4','avi')), fourcc, 30,(dmap.shape[1]+frame.shape[1],min_height), True)
                    
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
                    writer.write(concated)
                    if args is not None and args.get('silent',True):
                        print('silent')
                        encoded=cv2.imencode('.jpg', concated)[1].tobytes()
                        yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n\r\n')
                    else: cv2.imshow("Surveillence", concated)


                if log_count:
                    log_count_fcn(os.path.join(args['output'],'temp.csv'),len(list(objects)))

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
            params = {
                    'input': video_path,
                    'silent': True,
                    'write_output': True,
                }
            if args is None:
                args = params
            else: 
                args.update(params) 
            if args.get('log_counts',False):
                args['output']=output
            if args.get('live_data',False):
                args['queue']=QUEUE    

            for x in cls.model.forward_video(args):
                # queue.put_nowait(x)
                # QUEUE.put_nowait(x)
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





# import os,sys
# import cv2,imutils
# ROOT_PATH='C:\\Users\\PC\\Desktop\\PFE related\\applications\\CC App\\Developpement'
# VIDEO_PATH='C:\\Users\\PC\\Downloads\\Video\\hadj.mp4'
# model_type='SANet'
# sys.path.append(ROOT_PATH)
# import config
# from modelmanager import ModelManager
# ModelManager.set_base_path(config.FROZEN_MODELS_BASE_PATH)
# ModelManager.load_external_model(model_type)
# for x in ModelManager.process_video(VIDEO_PATH,args=dict(output=os.path.join(ROOT_PATH,'ressources','videos'))):
#     # cv2.imshow('frame22222',x)
#     pass

