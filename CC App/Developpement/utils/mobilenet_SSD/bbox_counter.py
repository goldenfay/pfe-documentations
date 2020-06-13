

import os,sys,glob,inspect
import multiprocessing
from multiprocessing import Queue
import pathos
from pathos.multiprocessing import ProcessingPool as Pool
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
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
			print('person found')
				# compute coordinates of the bounding box
			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY, endX, endY) = (
				int(startX), int(startY), int(endX), int(endY))
			
			rect = dlib.rectangle(startX, startY, endX, endY)

			if show_bbox:
				cv2.rectangle(frame,(startX,startY),(endX,endY),(255,0,0),thickness=4)
			# cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
	text='Persons : {}'.format(count)
	cv2.putText(frame, text, (10,frame.shape[1]-10),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)		
	return frame,detections.shape[2]


def process_video(net,vs,write_output=False,min_confidence=0.4,skip_frames=10,silent=False):
	
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
	
	(H, W) = frame.shape[:2]
	writer=None
	if write_output:
		fourcc = cv2.VideoWriter_fourcc(*"MP4V")
		writer = cv2.VideoWriter(os.path.join(currentdir,'output','output.mp4'), fourcc, 30,
			(W, H), True)

	fps = FPS().start()


	while True:
		frame = imutils.resize(frame, width=500)
		rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		(H, W) = frame.shape[:2]
		rects=[]
		frame=rgb_frame
		print('processing frame n° ',totalFrames)
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
						print('person found')
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
					# get the object(person) direction the difference between the y-coordinate of the current
					# object and the mean of it previous centroids
				y = [c[1] for c in tracked_obj.centroids]
				direction = centroid[1] - np.mean(y)
				tracked_obj.centroids.append(centroid)

				
				if not tracked_obj.counted:
					if direction < 0 and centroid[1] < H // 2:
						countUp += 1
						tracked_obj.counted = True
					elif direction > 0 and centroid[1] > H // 2:
						countDown += 1
						tracked_obj.counted = True
			trackableObjects[objectID] = tracked_obj

				# draw both the ID of the object and the centroid of the
			text = "Person {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
			#cv2.imshow("Frame", frame)

		info = [
		("Up", countUp),
		("Down", countDown)
	]

			# lShow infos on he frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)	
		
		if  writer is not None:
			writer.write(frame)

		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break

		totalFrames+=1
		fps.update()
		if silent:	
			encoded=cv2.imencode('.jpg', frame)[1].tobytes()
			yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n\r\n')
		else:	
			cv2.imshow("Frame", frame)

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

		def __init__(self, model_name):
			self.net=MobileSSD.load_net()
			

		@classmethod
		def load_net(cls):
			return  load_network()

		
		def forward(self,frames,confidence=0.4,show_bbox=True):
			frames=np.array(frames*255,dtype='uint8')
			img,count=process_frame(self.net,frames,show_bbox=show_bbox)
			
			return img,count
			jobs=[]
			# queue=[None for _ in range(len(frames))]
			queue=Queue()
			results_tab=[]

			process_fcn=lambda net,frame,conf,bbox_flag,index,tab:tab.insert(index,process_frame(net,frame,conf,bbox_flag) )
			
			for i,frame in enumerate(frames):
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
			
			global server
			if not args.get("input", False):
				video_path='ressources/videos/example_02.mp4'
				print("[INFO] Opening webcam...")
				vs = cv2.VideoCapture(video_path)
				
			else:
				print("[INFO] opening video file...")
				vs = cv2.VideoCapture(args["input"])
			
			if server is None:
				
				server=Flask(__name__)
				@server.route('/video_feed')
				def video_feed():
					return Response(process_video(self.net,vs,write_output=args['write_output'],silent=args['silent']),
									mimetype='multipart/x-mixed-replace; boundary=frame')
				# pool=multiprocessing.Pool(1)
				# p=pool.apply_async(run_simple,('localhost',4000,server,))
				# p.get()
				run_simple('localhost',4000,server,use_reloader=False,threaded=True)
				# pool=Pool(1)
				# pool.map(lambda :server.run(port=4000,threaded=True),[])
				# multiprocessing.Process(target=lambda :server.run(port=4000,threaded=True)).start()
				print('lkgjdflkgjdklfjgkjflkgjdflkj')

				# If it's a video capture
			if not args.get("input", False):
				vs.stop()

			else: # it's a webcam video stream
				vs.release()

			


if __name__=='__main__':
	pass
	# construct the argument parse and parse the arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-p", "--prototxt", required=True,
	# 	help="path to Caffe 'deploy' prototxt file")
	# ap.add_argument("-m", "--model", required=True,
	# 	help="path to Caffe pre-trained model")
	# ap.add_argument("-i", "--input", type=str,
	# 	help="path to optional input video file")
	# ap.add_argument("-o", "--output", type=str,
	# 	help="path to optional output video file")
	# ap.add_argument("-c", "--confidence", type=float, default=0.4,
	# 	help="minimum probability to filter weak detections")
	# ap.add_argument("-s", "--skip-frames", type=int, default=30,
	# 	help="# of skip frames between detections")
	# args = vars(ap.parse_args())

	

	

	
		
	# # initialize the video writer (we'll instantiate later if need be)
	# writer = None

	# # initialize the frame dimensions (we'll set them as soon as we read
	# # the first frame from the video)
	# W = None
	# H = None

	# # instantiate our centroid tracker, then initialize a list to store
	# # each of our dlib correlation trackers, followed by a dictionary to
	# # map each unique object ID to a TrackableObject
	# ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
	# trackers = []
	# trackableObjects = {}

	# # initialize the total number of frames processed thus far, along
	# # with the total number of objects that have moved either up or down
	# totalFrames = 0
	# totalDown = 0
	# totalUp = 0

	# # start the frames per second throughput estimator
	# fps = FPS().start()

	# # loop over frames from the video stream
	# while True:
	# 	# grab the next frame and handle if we are reading from either
	# 	# VideoCapture or VideoStream
	# 	frame = vs.read()
	# 	frame = frame[1] if args.get("input", False) else frame

	# 	# if we are viewing a video and we did not grab a frame then we
	# 	# have reached the end of the video
	# 	if args["input"] is not None and frame is None:
	# 		print('No frame captured')
	# 		break

	# 	# resize the frame to have a maximum width of 500 pixels (the
	# 	# less data we have, the faster we can process it), then convert
	# 	# the frame from BGR to RGB for dlib
	# 	frame = imutils.resize(frame, width=500)
	# 	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# 	# if the frame dimensions are empty, set them
	# 	if W is None or H is None:
	# 		(H, W) = frame.shape[:2]

	# 	# if we are supposed to be writing a video to disk, initialize
	# 	# the writer
	# 	if args["output"] is not None and writer is None:
	# 		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	# 		writer = cv2.VideoWriter(args["output"], fourcc, 30,
	# 			(W, H), True)

	# 	# initialize the current status along with our list of bounding
	# 	# box rectangles returned by either (1) our object detector or
	# 	# (2) the correlation trackers
	# 	status = "Waiting"
	# 	rects = []

	# 	# check to see if we should run a more computationally expensive
	# 	# object detection method to aid our tracker
	# 	if totalFrames % args["skip_frames"] == 0:
	# 		# set the status and initialize our new set of object trackers
	# 		status = "Detecting"
	# 		trackers = []

	# 		# convert the frame to a blob and pass the blob through the
	# 		# network and obtain the detections
	# 		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
	# 		net.setInput(blob)
	# 		detections = net.forward()

	# 		# loop over the detections
	# 		for i in np.arange(0, detections.shape[2]):
	# 			# extract the confidence (i.e., probability) associated
	# 			# with the prediction
	# 			confidence = detections[0, 0, i, 2]

	# 			# filter out weak detections by requiring a minimum
	# 			# confidence
	# 			if confidence > args["confidence"]:
	# 				# extract the index of the class label from the
	# 				# detections list
	# 				idx = int(detections[0, 0, i, 1])

	# 				# if the class label is not a person, ignore it
	# 				if CLASSES[idx] != "person":
	# 					continue

	# 				# compute the (x, y)-coordinates of the bounding box
	# 				# for the object
	# 				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
	# 				(startX, startY, endX, endY) = box.astype("int")
	# 				(startX, startY, endX, endY)=(int(startX),int( startY),int( endX),int( endY))
	# 				# construct a dlib rectangle object from the bounding
	# 				# box coordinates and then start the dlib correlation
	# 				# tracker
	# 				tracker = dlib.correlation_tracker()
	# 				rect = dlib.rectangle(startX, startY, endX, endY)
	# 				tracker.start_track(rgb, rect)

	# 				# add the tracker to our list of trackers so we can
	# 				# utilize it during skip frames
	# 				trackers.append(tracker)

	# 	# otherwise, we should utilize our object *trackers* rather than
	# 	# object *detectors* to obtain a higher frame processing throughput
	# 	else:
	# 		# loop over the trackers
	# 		for tracker in trackers:
	# 			# set the status of our system to be 'tracking' rather
	# 			# than 'waiting' or 'detecting'
	# 			status = "Tracking"

	# 			# update the tracker and grab the updated position
	# 			tracker.update(rgb)
	# 			pos = tracker.get_position()

	# 			# unpack the position object
	# 			startX = int(pos.left())
	# 			startY = int(pos.top())
	# 			endX = int(pos.right())
	# 			endY = int(pos.bottom())

	# 			# add the bounding box coordinates to the rectangles list
	# 			rects.append((startX, startY, endX, endY))

	# 	# draw a horizontal line in the center of the frame -- once an
	# 	# object crosses this line we will determine whether they were
	# 	# moving 'up' or 'down'
	# 	cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

	# 	# use the centroid tracker to associate the (1) old object
	# 	# centroids with (2) the newly computed object centroids
	# 	objects = ct.update(rects)

	# 	# loop over the tracked objects
	# 	for (objectID, centroid) in objects.items():
	# 		# check to see if a trackable object exists for the current
	# 		# object ID
	# 		to = trackableObjects.get(objectID, None)

	# 		# if there is no existing trackable object, create one
	# 		if to is None:
	# 			to = TrackableObject(objectID, centroid)

	# 		# otherwise, there is a trackable object so we can utilize it
	# 		# to determine direction
	# 		else:
	# 			# the difference between the y-coordinate of the *current*
	# 			# centroid and the mean of *previous* centroids will tell
	# 			# us in which direction the object is moving (negative for
	# 			# 'up' and positive for 'down')
	# 			y = [c[1] for c in to.centroids]
	# 			direction = centroid[1] - np.mean(y)
	# 			to.centroids.append(centroid)

	# 			# check to see if the object has been counted or not
	# 			if not to.counted:
	# 				# if the direction is negative (indicating the object
	# 				# is moving up) AND the centroid is above the center
	# 				# line, count the object
	# 				if direction < 0 and centroid[1] < H // 2:
	# 					totalUp += 1
	# 					to.counted = True

	# 				# if the direction is positive (indicating the object
	# 				# is moving down) AND the centroid is below the
	# 				# center line, count the object
	# 				elif direction > 0 and centroid[1] > H // 2:
	# 					totalDown += 1
	# 					to.counted = True

	# 		# store the trackable object in our dictionary
	# 		trackableObjects[objectID] = to

	# 		# draw both the ID of the object and the centroid of the
	# 		# object on the output frame
	# 		text = "ID {}".format(objectID)
	# 		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
	# 			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	# 		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# 	# construct a tuple of information we will be displaying on the
	# 	# frame
	# 	info = [
	# 		("Up", totalUp),
	# 		("Down", totalDown),
	# 		("Status", status),
	# 	]

	# 	# loop over the info tuples and draw them on our frame
	# 	for (i, (k, v)) in enumerate(info):
	# 		text = "{}: {}".format(k, v)
	# 		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
	# 			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# 	# check to see if we should write the frame to disk
	# 	if writer is not None:
	# 		writer.write(frame)

	# 	# show the output frame
	# 	cv2.imshow("Frame", frame)
	# 	key = cv2.waitKey(1) & 0xFF

	# 	# if the `q` key was pressed, break from the loop
	# 	if key == ord("q"):
	# 		break

	# 	# increment the total number of frames processed thus far and
	# 	# then update the FPS counter
	# 	totalFrames += 1
	# 	fps.update()

	# # stop the timer and display FPS information
	# fps.stop()
	# print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# # check to see if we need to release the video writer pointer
	# if writer is not None:
	# 	writer.release()

	# # if we are not using a video file, stop the camera video stream
	# if not args.get("input", False):
	# 	vs.stop()

	# # otherwise, release the video file pointer
	# else:
	# 	vs.release()

	# # close any open windows
	# cv2.destroyAllWindows()
