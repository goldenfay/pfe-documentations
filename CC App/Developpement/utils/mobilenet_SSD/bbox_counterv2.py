

import os,sys,glob,inspect
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse,time,datetime
import requests
import cv2
import dlib
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(currentdir)
from trackers.centroidtracker import CentroidTracker
from trackers.trackableobject import TrackableObject

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
	return cv2.dnn.readNetFromCaffe(protopath, caffepath)


def get_capture(video_path=None, webcam=False):
		
	if webcam:
		print("[INFO] starting video stream...")
		vs = VideoStream(src=0).start()
		time.sleep(2.0)

		
	else:
		print("[INFO] opening video file...")
		vs = cv2.VideoCapture(video_path)



	

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
	ap.add_argument("-s", "--skip-frames", type=int, default=15,
		help="# of skiped frames between  two expensive detections")
	args = vars(ap.parse_args())

		
		# load SSD model from disk
	print("[INFO] loading model...")
	if not args.get("prototxt",False)  or  not args.get("model",False):
		net=load_network()
	else:
		net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	if not args.get("input", False):
		video_path='videos/example_03.mp4'
		print("[INFO] Opening webcam...")
		# vs = VideoStream(src=0).start()
		vs = cv2.VideoCapture(video_path)
		
	else:
		print("[INFO] opening video file...")
		print(args['input'])
		vs = cv2.VideoCapture(args["input"])	

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
	if args['output']:
		fourcc = cv2.VideoWriter_fourcc(*"MP4V")
		writer = cv2.VideoWriter(os.path.join(currentdir,'output','output2.mp4'), fourcc, 30,
			(W, H), True)

	fps = FPS().start()


	while True:
		frame = imutils.resize(frame, width=500)
		rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		(H, W) = frame.shape[:2]
		rects=[]
		if totalFrames % args['skip_frames']== 0: # Use model detection, expensive process
				
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
					if confidence > args['confidence']:
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
		cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
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
		cv2.imshow("Frame", frame)
		
		if  writer is not None:
			writer.write(frame)

		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break

		totalFrames+=1
		fps.update()
		frame = vs.read()
			#VideoStream returns a frame, VideoCapture returns a tuple
		frame = frame[1] if len(frame)>1 else frame
		if frame is None:
			break
	
	

	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	
	if writer is not None:
		writer.release()

		# If it's a video capture
	if not args.get("input", False):
		vs.stop()

	else: # it's a webcam video stream
		vs.release()
		

		# close any open windows
	cv2.destroyAllWindows()	