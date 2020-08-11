import os,sys,glob,inspect,argparse,time,cv2,imutils,dlib
import multiprocessing
from multiprocessing import Queue
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
from trackers.centroidtracker import CentroidTracker
from trackers.trackableobject import TrackableObject


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# initialize the list of class labels MobileNet SSD was trained to
	# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]


def load_network():
	# load our serialized model from disk
	print("[INFO] loading model...")
	protopath=os.path.join(currentdir,'mobilenet_ssd','MobileNetSSD_deploy.prototxt')
	caffepath=os.path.join(currentdir,'mobilenet_ssd','MobileNetSSD_deploy.caffemodel')
	return cv2.dnn.readNetFromCaffe(protopath, caffepath)


def get_capture(video_path=None, webcam=False):
	# if a video path was not supplied, grab a reference to the webcam
	if webcam:
		print("[INFO] starting video stream...")
		vs = VideoStream(src=0).start()
		time.sleep(2.0)

	# otherwise, grab a reference to the video file
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
		# extract the confidence (i.e., probability) associated
		# with the prediction
		confidence = detections[0, 0, i, 2]
		if CLASSES[int(detections[0, 0, i, 1])] == "person": count+=1
		# filter out weak detections by requiring a minimum
		# confidence
		if confidence > min_conf:
			# extract the index of the class label from the
			# detections list
			idx = int(detections[0, 0, i, 1])

			# if the class label is not a person, ignore it
			if CLASSES[idx] != "person":
				continue
			print('person found')
			# compute the (x, y)-coordinates of the bounding box
			# for the object
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
		fourcc = cv2.VideoWriter_fourcc(*"MP4")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)

	fps = FPS().start()


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
					print(confidence)
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
			cv2.imshow("Frame", frame)

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

		if not silent:	
			cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break

		totalFrames+=1
		fps.update()
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	
	if writer is not None:
		writer.release()


def process_to_queue(net,frame,conf,bbox_flag,index,queue):
	queue.put({'idx':str(index),'val':process_frame(net,frame,conf,bbox_flag)})



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