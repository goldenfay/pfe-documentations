import os,sys,inspect
import numpy as np
import cv2
import imutils
from imutils.video import FPS

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(currentdir)
from utils.detection_model import DetectionModel
from detector import *

class YOLO(DetectionModel):

    def __init__(self,use_tiny=False):
        self.use_tiny=use_tiny
        self.net=YOLO.load_net(use_tiny)
        self.init_params()

    @classmethod
    def load_net(cls,use_tiny):
        cls.config = read_config(os.path.join(currentdir,'config.ini'))

            # Load the trained network
        (net, cls.ln, cls.LABELS) = load_network(os.path.join(currentdir,'yolo-coco'),tiny_version=use_tiny)
        return net

    def forward(self,frame):
        frame=np.array(frame*255,dtype='uint8')
        frame = imutils.resize(frame, width=500)
        # frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # while True:
        #     cv2.imshow("Frame",frame)
        #     key = cv2.waitKey(1) & 0xFF

        #     if key == ord("q"):
        #         break
        (H, W) = frame.shape[:2]
            # Feed frame to network
        layerOutputs = forward_detection(frame, self.net, YOLO.ln)
            # Obtain detected objects, including cof levels and bounding boxes
        (idxs, classIDs, boxes, confidences) = get_detected_items(layerOutputs, self.confidence, self.threshold,
                                                                  W, H)
        print(len(layerOutputs))
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(boxes), 3), dtype="uint8")
        npeople=len(boxes)

        for i,box in enumerate(boxes):
            print(box)
            (x, y, w, h) = (box[0], box[1], box[2], box[3])
            if YOLO.config['OUTPUT']['ShowPeopleBoxes'] == "yes":
                color = (COLORS[i,0],COLORS[i,1],COLORS[i,2])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (237, 213, 33), 4)
                text = "Person {}".format(i)
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (237, 213, 33), 2)
            else: print('kjfghsdfkjghkjfdhgkjdfhgkjdhf')
        text = "Persons found : {}".format(npeople)
        cv2.putText(frame, text, (10,frame.shape[0]- 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 4)

        return frame,npeople                              

    def forward_video(self,args):
        if args is None:
            args={
                'silent':True
            }
        for f in process_video(args=args):
            yield f
                
    def init_params(self):
        self.confidence = float(YOLO.config['NETWORK']['Confidence'])
        self.threshold = float(YOLO.config['NETWORK']['Threshold'])
        print('Confidence : ',self.confidence,' Threshold : ',self.threshold)

