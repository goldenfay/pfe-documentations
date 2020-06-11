
import os,sys,glob,inspect,argparse,time,cv2,imutils,dlib
import multiprocessing
from multiprocessing import Queue
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(currentdir)
from trackers.centroidtracker import CentroidTracker
from trackers.trackableobject import TrackableObject

from bbox_counter import *

if __name__=='__main__':
		# Define program arguments
	ap = argparse.ArgumentParser(description="Mobilenet SSD detector. Copyrights @Ihasel")
	ap.add_argument("-p", "--prototxt", 
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", 
		help="path to Caffe pre-trained model")
	ap.add_argument("-i", "--input", type=str,
		help="path to optional input video file")
	ap.add_argument("-o", "--output", type=str,
		help="path to optional output video file")
	ap.add_argument("-c", "--confidence", type=float, default=0.4,
		help="minimum probability to filter detections")
	ap.add_argument("-s", "--skip-frames", type=int, default=30,
		help="# of skiped frames between  two expensive detections")
	args = vars(ap.parse_args())


		
		# load SSD model from disk
	print("[INFO] loading model...")
	if not args.get("prototxt",False)  or  not args.get("model",False):
		net=load_network()
	else:
		net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

		
	if not args.get("input", False):
		video_path='ressources/videos/example_02.mp4'
		print("[INFO] Opening webcam...")
		# vs = VideoStream(src=0).start()
		vs = cv2.VideoCapture(video_path)
		
	else:
		print("[INFO] opening video file...")
		vs = cv2.VideoCapture(args["input"])

	process_video(net,vs,write_output=False)

		# If it's a video capture
	if not args.get("input", False):
		vs.stop()

	else: # it's a webcam video stream
		vs.release()

		# close any open windows
	cv2.destroyAllWindows()	