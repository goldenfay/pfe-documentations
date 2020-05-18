import numpy as np
import imutils
import time
import dlib
import cv2
import sys
sys.path.append('.')


class DetectionModel:
    def __init__(self,model_name):
        pass
    #    if model_name=='mobileSSD':
    #        self.net=ssd_module.MobileSSD.load_net()
    #    else:
    #        self.net= yolo_module.YOLO.load_net()
    


    def forward(self,frame):
        pass
        # if isinstance(self.net,ssd_module.MobileSSD):
        #    self.net=ssd_module.MobileSSD.forward(frame)
        # else:
        #    self.net= yolo_module.YOLO.load_net()
