

import os,sys,glob,inspect
import multiprocessing
from multiprocessing import Queue
# import pathos
# from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing.pool import ThreadPool as Pool
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import pandas as pd
import argparse,time,datetime
import requests
import cv2
import dlib
from werkzeug.serving import run_simple
from flask import Flask, Response

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(currentdir)
from trackers.centroidtracker import CentroidTracker
from trackers.trackableobject import TrackableObject

from utils.detection_model import DetectionModel

modelfolder=os.path.join(currentdir,'mobilenet_ssd')
if not os.path.exists(modelfolder):
	os.makedirs(modelfolder)
if not os.path.exists(os.path.join(currentdir,'output')):
	os.makedirs(os.path.join(currentdir,'output'))

# initialize the MobilenetSSD list of class labels 
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

QUEUE=None
server=None

def download_if_not_present(url, file_name):

	if not os.path.exists(file_name):
		with open(file_name, "wb") as f:
			response = requests.get(url, stream=True)
			total_length = response.headers.get('content-length')
			if total_length is None:
				# no content length header
				f.write(response.content)
			else:
				print_file_name = "..." + os.path.basename(file_name) 
				print_file_name = "{:<20}".format(print_file_name)
				downloaded = 0
				total_length = int(total_length)
				for data in response.iter_content(chunk_size=4096):
					downloaded += len(data)
					f.write(data)
					percentage = min(int(100 * downloaded / total_length), 100)
					progress = min(int(50 * downloaded / total_length), 50)
					sys.stdout.write("\rDownloading {} [{} {}] {}%".format(print_file_name, '=' * progress,
																			' ' * (50-progress), percentage))
					sys.stdout.flush()
				sys.stdout.write("\n")
				sys.stdout.flush()
		print('Download finished.')

def load_network():
		# Check if required files (.prototxt and .caffemodel) exists. If not download them
	caffepath = os.path.join(modelfolder, "MobileNetSSD_deploy.caffemodel")
	download_if_not_present("https://github.com/djmv/MobilNet_SSD_opencv/blob/master/MobileNetSSD_deploy.caffemodel?raw=true", caffepath)
	protopath = os.path.join(modelfolder, "MobileNetSSD_deploy.prototxt")
	download_if_not_present("https://github.com/djmv/MobilNet_SSD_opencv/blob/master/MobileNetSSD_deploy.prototxt?raw=true", protopath)
		# load our serialized model from disk
	print("[INFO] loading model...")
	return cv2.dnn.readNetFromCaffe(protopath, caffepath)


def get_capture(video_path=None, webcam=False):
		
	if webcam:
		print("[INFO] starting video stream...")
		vs = VideoStream(src=0).start()
		time.sleep(2.0)

		
	else:
		print("[INFO] opening video file...")
		vs = cv2.VideoCapture(video_path)


def process_frame(net,frame,min_conf=0.4,show_bbox=True):
	if net is None:
		net=load_network()
	frame = imutils.resize(frame, width=500)
	rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	(H, W) = frame.shape[:2]
	count=0
		# convert the frame to a blob and pass the blob through the
		# network and obtain the detections
	blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
	net.setInput(blob)
	detections = net.forward()
	if detections.shape[2]>0: print('Found detections')
	else:
		print('No detection found')
		# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		
		confidence = detections[0, 0, i, 2]
		if CLASSES[int(detections[0, 0, i, 1])] == "person": count+=1
			# filter weak detection (those under min confidence)
		if confidence > min_conf:
				# extract the index of the class label 
			idx = int(detections[0, 0, i, 1])

			if CLASSES[idx] != "person":
				continue
				# compute coordinates of the bounding box
			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY, endX, endY) = (
				int(startX), int(startY), int(endX), int(endY))
			

			if show_bbox:
				cv2.rectangle(frame,(startX,startY),(endX,endY),(255,0,0),thickness=4)
			# cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
	text='Persons : {}'.format(count)
	cv2.putText(frame, text, (10,frame.shape[0]-10),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)		
	return frame,count


def process_video(net,vs,write_output=False,min_confidence=0.4,skip_frames=10,silent=False,args=None):
	print('[INFO] Initializing ...')
		#Split regions params
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
	queue=args is not None and args.get('queue',None)

	ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
	trackers = []
	trackableObjects = {}
	totalFrames=0
	countUp=0
	countDown=0
	frame = vs.read()
		#VideoStream returns a frame, VideoCapture returns a tuple
	frame = frame[1] if len(frame)>1 else frame

	if frame is None:
		raise Exception('[FATAL] Cannot read video stream')
	frame = imutils.resize(frame, width=500)
	(H, W) = frame.shape[:2]
	writer=None
	if write_output:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		if args.get('output',False):
			writer = cv2.VideoWriter(os.path.join(args['output'],os.path.basename(args['input'].replace('.mp4','.avi'))), fourcc, 30,
			(W, H), True)
		else:
			writer = cv2.VideoWriter(os.path.join(currentdir,'output',os.path.basename(args['input'].replace('.mp4','.avi'))), fourcc, 30,
			(W, H), True)

	fps = FPS().start()

	print('[INFO] Video in process ...')

	while True:
		frame = imutils.resize(frame, width=500)
		rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		(H, W) = frame.shape[:2]
		rects=[]
		if totalFrames % skip_frames== 0: # Use model detection, expensive process
				
				trackers = []

					# convert the frame to a blob and pass it through the net
			
				blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
				net.setInput(blob)
				detections = net.forward()
		
					# loop over the detections
				for i in np.arange(0, detections.shape[2]):
						# extract the confidence (i.e., probability) associated
						# with the prediction
					confidence = detections[0, 0, i, 2]
						# filter detections  under minimum confidence
					if confidence > min_confidence:
							# extract the index of the class label from the detection
						idx = int(detections[0, 0, i, 1])

						if CLASSES[idx] != "person":
							continue
							# compute the coordinates of the bounding box of the object
						box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
						(startX, startY, endX, endY) = box.astype("int")
						(startX, startY, endX, endY) = (
							int(startX), int(startY), int(endX), int(endY))
							# construct a dlib rectangle object from the bbox and start tracking
							
						tracker = dlib.correlation_tracker()
						rect = dlib.rectangle(startX, startY, endX, endY)
						tracker.start_track(rgb_frame, rect)

						trackers.append(tracker)

		
		else:	# Else  use tracking to get new positions of detected objects 

					# loop over the trackers
				for tracker in trackers:
				

						# update the tracker and get the updated position
					tracker.update(rgb_frame)
					pos = tracker.get_position()

					startX = int(pos.left())
					startY = int(pos.top())
					endX = int(pos.right())
					endY = int(pos.bottom())

					rects.append((startX, startY, endX, endY))


			# updates tracked objects list
		objects = ct.update(rects)
			# loop over the tracked objects
		for (objectID, centroid) in objects.items():
				# check to see if the trackable object with this ID already exists 
			tracked_obj = trackableObjects.get(objectID, None)

				# if there is no existing trackable object, create one
			if tracked_obj is None:
				tracked_obj = TrackableObject(objectID, centroid)

			else:
				tracked_obj.centroids.append(centroid)
				

				if show_regions:
						# determines wether direction will be based on x or y coordinate (horizontal/vertical splitting)
					coord_ref=1 if horizontal_splited else 0

						# get the object(person) direction the difference between the y/x-coordinate of the current
						# object and the mean of it previous centroids
					coord = [c[coord_ref] for c in tracked_obj.centroids]
					direction = centroid[coord_ref] - np.mean(coord)
				

					if not tracked_obj.counted:
					
					
						if direction < 0 and centroid[coord_ref] < line_eq(centroid[coord_ref]):
							countUp += 1
							tracked_obj.counted = True
						elif direction > 0 and centroid[coord_ref] > line_eq(centroid[coord_ref]):
							countDown += 1
							tracked_obj.counted = True
						
			trackableObjects[objectID] = tracked_obj

				# draw both the ID of the object and the centroid of the
			text = "Person {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
		
		
		info=[("Current" , len(list(objects)))]
		if show_regions:
			info += [
			("Up" if horizontal_splited else "Right", countUp),
			("Down" if horizontal_splited else "Left", countDown)
			]

				# Show infos on he frame
			for (i, (k, v)) in enumerate(info):
				text = "{}: {}".format(k, v)
				cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
			cv2.line(frame,(0,line_eq(0)),(frame.shape[1],line_eq(frame.shape[1])),(0,200,0),5)				
		
		if  writer is not None:
			writer.write(frame)

		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break
		totalFrames+=1
		fps.update()
		if silent:	
			encoded=cv2.imencode('.jpg', frame)[1].tobytes()
			# QUEUE.put( b'--frame\r\n'
			# 	b'Content-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n\r\n')
			yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n\r\n')
		else:	
			cv2.imshow("Frame", frame)
		if log_count:
			log_count_fcn(os.path.join(args['output'],'temp.csv'),len(list(objects)))
		if queue is not None:
			queue.put_nowait({'timestamp': pd.Timestamp(datetime.datetime.now()),'value':len(list(objects))})
		frame = vs.read()
			#VideoStream returns a frame, VideoCapture returns a tuple
		frame = frame[1] if len(frame)>1 else frame
		if frame is None:
			print('[Warning] Red a None frame')
			break	

	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	
	if writer is not None:
		writer.release()

def process_to_queue(net,frame,conf,bbox_flag,index,queue):
	queue.put({'idx':str(index),'val':process_frame(net,frame,conf,bbox_flag)})

def stream_video(args):
		
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

	process_video(net,vs,write_output=args['write_output'],silent=args['silent'])

		# If it's a video capture
	if not args.get("input", False):
		vs.stop()

	else: # it's a webcam video stream
		vs.release()

	if not args.get("silent", False):
			# close any open windows
		cv2.destroyAllWindows()	

class MobileSSD(DetectionModel):

		def __init__(self):
			self.net=MobileSSD.load_net()
			

		@classmethod
		def load_net(cls):
			return  load_network()

		
		def forward(self,image,confidence=0.4,show_bbox=True):
			image=np.array(image*255,dtype='uint8')
			img,count=process_frame(self.net,image,show_bbox=show_bbox)
			
			return img,count

			jobs=[]
			# queue=[None for _ in range(len(frames))]
			queue=Queue()
			results_tab=[]

			process_fcn=lambda net,frame,conf,bbox_flag,index,tab:tab.insert(index,process_frame(net,frame,conf,bbox_flag) )
			
			for i,frame in enumerate(image):
				frame=np.array(frame*255,dtype='uint8')
				# print(type(frame), frame[3:5,2:4,1])
				
				process=multiprocessing.Process(target=process_to_queue,args=(None,frame,confidence,show_bbox,i,queue))
				jobs.append(process)

			for i in range(len(jobs)):
				jobs[i].start()
			print('going to join them ')
				# img,count=process_frame(self.net,frame,show_bbox=show_bbox)
			for i in range(len(jobs)):
				jobs[i].join()
				print('joined')
			while not queue.empty():
				print('yep')
				x=queue.get()
				results_tab.insert(int(x['idx']),x['val'])	
			print(results_tab)		
			return results_tab	

		def forward_video(self,args):
			
			global server,QUEUE
			if not args.get("input", False):
				video_path='ressources/videos/example_02.mp4'
				print("[INFO] Opening webcam...")
				vs = cv2.VideoCapture(video_path)
				
			else:
				print("[INFO] opening video file...")
				vs = cv2.VideoCapture(args["input"])
			
			# process_video(self.net,vs,write_output=args['write_output'],silent=args['silent'])
			for f in process_video(self.net,vs,write_output=args['write_output'],silent=args['silent'],args=args):
				
				# QUEUE.put_nowait(f)
				yield f
				

				# If it's a video capture
			if not args.get("input", False):
				vs.stop()

			else: # it's a webcam video stream
				vs.release()

			


if __name__=='__main__':
	
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt",
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model",
		help="path to Caffe pre-trained model")
	ap.add_argument("-i", "--input", type=str,
		help="path to optional input video file")
	ap.add_argument("-o", "--output", type=str,
		help="path to optional output video file")
	ap.add_argument("-c", "--confidence", type=float, default=0.4,
		help="minimum probability to filter weak detections")
	ap.add_argument("-s", "--skip-frames", type=int, default=30,
		help="# of skip frames between detections")
	args = vars(ap.parse_args())

	if not args.get("input", False):
				video_path='ressources/videos/example_02.mp4'
				print("[INFO] Opening webcam...")
				vs = cv2.VideoCapture(video_path)
				
	else:
		print("[INFO] opening video file...")
		vs = cv2.VideoCapture(args["input"])

	net=load_network()		
	process_video(net,vs,write_output=False,silent=False)	
	if not args.get("input", False):
				vs.stop()

	else: # it's a webcam video stream
		vs.release()	


	
