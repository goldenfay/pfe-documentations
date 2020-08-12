
import argparse,time,os,sys,datetime,glob,inspect
import configparser
import csv
import requests

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import dlib
from imutils.video import VideoStream
from imutils.video import FPS


currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
sys.path.append(currentdir)
from trackers.trackableobject import TrackableObject
from trackers.centroidtracker import CentroidTracker


if not os.path.exists(os.path.join(currentdir, 'yolo-coco')):
    os.makedirs(os.path.join(currentdir, 'yolo-coco'))
if not os.path.exists(os.path.join(currentdir, 'output')):
    os.makedirs(os.path.join(currentdir, 'output'))

def define_args():

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Configuration file")
    return vars(ap.parse_args())

def read_config(filename):

    print("[INFO] Reading config: {}".format(filename))
    if not os.path.isfile(filename):
        print("[ERROR] Config file \"{}\" not found.".format(filename))
        exit()
    cfg = configparser.ConfigParser()
    cfg.read(filename)
    return cfg

def download_if_not_present(url, file_name):

    if not os.path.exists(file_name):
        with open(file_name, "wb") as f:
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')
            if total_length is None:
                # no content length header
                f.write(response.content)
            else:
                print_file_name = "..." + \
                    file_name[-17:] if len(file_name) > 20 else file_name
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

def load_network(network_folder, tiny_version=False):

    labelspath = os.path.sep.join([network_folder, "coco.names"])
    download_if_not_present(
        "https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true", labelspath)
    if not os.path.isfile(labelspath):
        print("[ERROR] Network: Labels file \"{}\" not found.".format(labelspath))
        exit()
    weights_name = "yolov3.weights" if not tiny_version else "yolov3-tiny.weights"
    weightspath = os.path.sep.join([network_folder, weights_name])
    download_if_not_present(
        "https://pjreddie.com/media/files/"+weights_name, weightspath)
    if not os.path.isfile(weightspath):
        print("[ERROR] Network: Weights file \"{}\" not found.".format(weightspath))
        exit()
    config_name = "yolov3.cfg" if not tiny_version else "yolov3-tiny.cfg"
    configpath = os.path.sep.join([network_folder, config_name])
    download_if_not_present(
        "https://github.com/pjreddie/darknet/blob/master/cfg/"+config_name+"?raw=true", configpath)
    if not os.path.isfile(configpath):
        print("[ERROR] Network: Configuration file \"{}\" not found.".format(configpath))
        exit()

    print("[INFO] loading YOLO from disk...")
    labels = open(labelspath).read().strip().split("\n")
    network = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    names = network.getLayerNames()
    names = [names[i[0] - 1] for i in network.getUnconnectedOutLayers()]
    return network, names, labels


def get_capture(video_path=None, webcam=False, webcam_width=640, webcam_height=480):

    if webcam:
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        vs.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
        vs.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
        time.sleep(2.0)

    else:
        print("[INFO] opening video file...")
        vs = cv2.VideoCapture(video_path)
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return vs, width, height

def log_count(filename, n):

    f = open(filename, "a")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    line = "{} , {}\n".format(timestamp, n)
    f.write(line)
    f.close()

def get_videowriter(outputfile, width, height, frames_per_sec=30):

        # Initialise the writer
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_writer = cv2.VideoWriter(
        outputfile, fourcc, frames_per_sec, (width, height), True)
    return video_writer, frames_per_sec


def save_frame(video_writer, new_frame, count=1):

    for _ in range(0, count):
        video_writer.write(new_frame)

def read_existing_data(filename):

    times = []
    values = []
    if os.path.isfile(filename):
        with open(filename) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                times.append(datetime.datetime.strptime(
                    row[0], "%Y%m%d_%H-%M-%S "))
                values.append(int(row[1]))
    dataframe = pd.DataFrame()
    dataframe['timestamp'] = pd.Series(dtype='datetime64[ns]')
    dataframe['value'] = pd.Series(dtype=np.int32)
    dataframe['timestamp'] = times
    dataframe['value'] = values
    dataframe.set_index('timestamp', inplace=True)
    return dataframe


def forward_detection(image, network, layernames):

    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    start2 = time.time()
    network.setInput(blob)
    outputs = network.forward(layernames)
    end2 = time.time()
    print("[INFO] YOLO  took      : %2.1f sec" % (end2-start2))
    return outputs




def get_detected_items(layeroutputs, confidence_level, threshold, img_width, img_height):

    detected_boxes = []
    detection_confidences = []
    detected_classes = []
    for output in layeroutputs:
            # loop over detections
        for detection in output:
                # extract the class ID and confidence 
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]

                # filter weak predictions 
            if confidence > confidence_level and classid == 0:
            # if  classid == 0:
                print('\t Person found')
                    # scale the bounding box coordinates back relative to the size of the image
                box = detection[0:4] * \
                    np.array([img_width, img_height, img_width, img_height])
                (center_x, center_y, width, height) = box.astype("int")

                    # get coordinates of the top left point (start point) of the box
                top_x = int(center_x - (width / 2))
                top_y = int(center_y - (height / 2))

                    # update our list of bounding box coordinates, confidences, and class IDs
                detected_boxes.append([top_x, top_y, int(width), int(height)])
                detection_confidences.append(float(confidence))
                detected_classes.append(classid)

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indexes = cv2.dnn.NMSBoxes(
        detected_boxes, detection_confidences, confidence_level, threshold)

    return indexes, detected_classes, detected_boxes, detection_confidences


def process_video(config=None, args=None):
    print('fkjglfkjgdfmlkjgfldkjgflkdj')
    if config is None:
        print('[INFO] reading config file.')
        config_file = os.path.join(currentdir, 'config.ini')
        config = read_config(config_file)

        # Load the trained network
    (net, ln, LABELS) = load_network(os.path.join(currentdir,'yolo-coco'), tiny_version=True)
        # Get VideoStream if using webcam, or VideoCapture if using a video
    webcam = (config['READER']['Webcam'] == "yes")
    if webcam:
        cam_width,cam_height = int(config['READER']['Width']),int(config['READER']['Height'])
        video_path=None
    else:
        if args is not None and args.get('input',None) is not None:
            video_path=args['input']

        else: video_path=os.path.sep.join([currentdir]+config['READER']['Filename'].split('/'))
    (vs, cam_width, cam_height) = get_capture(webcam=webcam,video_path=video_path)

        # get params from config file
    showpeopleboxes = (config['OUTPUT']['ShowPeopleBoxes'] == "yes")
    showallboxes = (config['OUTPUT']['ShowAllBoxes'] == "yes")
    blurpeople = (config['OUTPUT']['BlurPeople'] == "yes")
    realspeed = (config['OUTPUT']['RealSpeed'] == "yes")
    nw_confidence = float(config['NETWORK']['Confidence'])
    nw_threshold = float(config['NETWORK']['Threshold'])
    countfile = config['OUTPUT']['Countfile']
    save_video = (config['OUTPUT']['SaveVideo'] == "yes")
    show_graphs = (config['OUTPUT']['ShowGraphs'] == "yes")
    print_ascii = (config['OUTPUT']['PrintAscii'] == "yes")
    SkipFrames = int(config['READER']['SkipFrames'])

        #get params from additional passed args
    silent=args['silent'] if args is not None and 'silent' in args else False

        # initialize a list of colors to represent each  class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    # Initialise video ouptut writer
    if save_video:
        (writer, fps) = get_videowriter(config['OUTPUT']['Filename'], cam_width, cam_height,
                                        int(config['OUTPUT']['FPS']))
    else:
        (writer, fps) = (None, 0)

    if not silent:
            # Create output windows, but limit on 1440x810
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Video', min(cam_width, 1440), min(cam_height, 810))
        cv2.resizeWindow('Video', min(cam_width, 640), min( cam_height, 360))
        cv2.moveWindow('Video', 0, 0)

        # Create plots if the flag is specified to True
    if show_graphs:
        plt.ion()
        plt.figure(num=None, figsize=(8, 7), dpi=80,
                   facecolor='w', edgecolor='k')
        df = read_existing_data(countfile)
    else:
        df = None

        #Initialize tracking and counting variables
    cent_tracker = CentroidTracker(maxDisappeared=50, maxDistance=90)
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}
    totalFrames=0
    countUp=0
    countDown=0   

        # Grab first frame to test if Stream is ok
    frame = vs.read()
        # VideoStream returns a frame, VideoCapture returns a tuple
    frame = frame[1] if len(frame) > 1 else frame

    if frame is None:
            raise Exception('[FATAL] Cannot read video stream')

    fps = FPS().start()
    totalFrames = 0
        # loop while true
    while True:
        (H, W) = frame.shape[:2]
        rects = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start = time.time()
        if totalFrames % SkipFrames== 0: # Use model detection, expensive process
				
                trackers = []
                    # Feed frame to network
                detections = forward_detection(frame, net, ln)
                    # Extract bounding boxes and related confidences from the detections
                (idxs, classIDs, boxes, confidences) = get_detected_items(detections, nw_confidence, nw_threshold, cam_width, cam_height)

                    # loop over bounding boxes
                for box in boxes:
                            
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(box[0], box[1], box[0]+box[2],box[1]+ box[3])
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

            # Show infos on he frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)	

            # If we should write the resulted video, then write the treated frame
        if  writer is not None:
            writer.write(frame)


        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        totalFrames+=1
        fps.update()

        if silent:	# if silent is True, we should return the frame to display it in other supplies
            encoded=cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n\r\n')
        else:	# Otherwise, we display it in the standard openCV window
            cv2.imshow("Video", frame)
     
            # Read next frame
        frame = vs.read()
            #VideoStream returns a frame, VideoCapture returns a tuple
        frame = frame[1] if len(frame)>1 else frame
        if frame is None:
            print('[Warning] Red a None frame')
            break

    print("[INFO] cleaning up...")
    fps.stop()
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # release the file pointers
    if save_video:
        writer.release()
    vs.release()
    cv2.destroyAllWindows()


def show_plots(data):
    """
    Show the graphs with historical data
    :param data: dataframe
    :return:
    """
    # Awful code to create new dataframes each time the graph is shown
    df_1w = data[data.index >= pd.datetime.now() - pd.Timedelta('7D')]
    df_1d = df_1w[df_1w.index >= pd.datetime.now() - pd.Timedelta('24H')]
    df_8h = df_1d[df_1d.index >= pd.datetime.now() - pd.Timedelta('8H')]
    df_2h = df_8h[df_8h.index >= pd.datetime.now() - pd.Timedelta('2H')]
    # Resample to smooth the long running graphs
    df_1w = df_1w.resample('1H').max()
    df_1d = df_1d.resample('15min').max()

    plt.gcf().clear()

    plt.subplot(2, 2, 1)
    plt.plot(df_1w.index.tolist(), df_1w['value'].tolist())
    plt.title("Laatste week")
    plt.ylabel("Personen")
    plt.xlabel("Tijdstip")

    plt.subplot(2, 2, 2)
    plt.plot(df_1d.index.tolist(), df_1d['value'].tolist())
    plt.title("Afgelopen 24 uur")
    plt.ylabel("Personen")
    plt.xlabel("Tijdstip")

    plt.subplot(2, 2, 3)
    plt.plot(df_8h.index.tolist(), df_8h['value'].tolist())
    plt.title("Afgelopen 8 uur")
    plt.ylabel("Personen")
    plt.xlabel("Tijdstip")

    plt.subplot(2, 2, 4)
    plt.plot(df_2h.index.tolist(), df_2h['value'].tolist())
    plt.title("Afgelopen 2 uur")
    plt.ylabel("Personen")
    plt.xlabel("Tijdstip")

    plt.gcf().autofmt_xdate()
    plt.show()

if __name__ == '__main__':
    for x in process_video():
        pass
  
    
    #     # construct the argument parse and parse the arguments
    # args = define_args()
    # config = read_config(args["config"])

    #     # Load the trained network
    # (net, ln, LABELS) = load_network(config['NETWORK']['Path'],tiny_version=True)
    # webcam = (config['READER']['Webcam'] == "yes")
    # if webcam:
    #     cam_id = int(config['READER']['WebcamID'])
    #     cam_width = int(config['READER']['Width'])
    #     cam_height = int(config['READER']['Height'])
    #     (cam, W, H) = get_capture(webcam=True,webcam=cam_width,webcam_height=cam_height)
    # else:
    #     (cam, cam_width, cam_height) = get_capture(video_path=config['READER']['Filename'])

    #     # get params from config file

    # showpeopleboxes = (config['OUTPUT']['ShowPeopleBoxes'] == "yes")
    # showallboxes = (config['OUTPUT']['ShowAllBoxes'] == "yes")
    # blurpeople = (config['OUTPUT']['BlurPeople'] == "yes")
    # realspeed = (config['OUTPUT']['RealSpeed'] == "yes")
    # nw_confidence = float(config['NETWORK']['Confidence'])
    # nw_threshold = float(config['NETWORK']['Threshold'])
    # countfile = config['OUTPUT']['Countfile']
    # save_video = (config['OUTPUT']['SaveVideo'] == "yes")
    # show_graphs = (config['OUTPUT']['ShowGraphs'] == "yes")
    # print_ascii = (config['OUTPUT']['PrintAscii'] == "yes")
    # SkipFrames = int(config['READER']['SkipFrames'])

    #     # initialize a list of colors to represent each  class label
    # np.random.seed(42)
    # COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    #     # Initialise video ouptut writer
    # if save_video:
    #     (writer, fps) = get_videowriter(config['OUTPUT']['Filename'], cam_width, cam_height,
    #                                     int(config['OUTPUT']['FPS']))
    # else:
    #     (writer, fps) = (None, 0)

    #     # Create output windows, but limit on 1440x810
    # cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    # # cv2.resizeWindow('Video', min(cam_width, 1440), min(cam_height, 810))
    # cv2.resizeWindow('Video', min(cam_width, 640), min(cam_height, 360))
    # cv2.moveWindow('Video', 0, 0)
    #     # Create plot
    # if show_graphs:
    #     plt.ion()
    #     plt.figure(num=None, figsize=(8, 7), dpi=80, facecolor='w', edgecolor='k')
    #     df = read_existing_data(countfile)
    # else:
    #     df = None
    # fps = FPS().start()
    #     # loop while true
    # while True:
    #     start = time.time()
    #         # read the next frame from the webcam
    #         # make sure that buffer is empty by reading specified amount of frames
    #     for _ in (0, SkipFrames):
    #         (grabbed, frame) = cam.read()  # type: (bool, np.ndarray)
    #     if not grabbed:
    #         break
    #         # Feed frame to network
    #     layerOutputs = forward_detection(frame, net, ln)
    #         # Obtain detected objects, including cof levels and bounding boxes
    #     (idxs, classIDs, boxes, confidences) = get_detected_items(layerOutputs, nw_confidence, nw_threshold,
    #                                                               cam_width, cam_height)

    #         # Update frame with recognised objects
    #     frame, npeople = update_frame(frame, idxs, classIDs, boxes, confidences, COLORS, LABELS, showpeopleboxes,
    #                                   blurpeople, showallboxes)
    #     log_count(countfile, npeople)

    #         # Show frame with bounding boxes on screen
    #     cv2.imshow('Video', frame)

    #     if show_graphs:
    #             # Add row to panda frame
    #         new_row = pd.DataFrame([[npeople]], columns=["value"], index=[pd.to_datetime(datetime.datetime.now())])
    #         df = pd.concat([df, pd.DataFrame(new_row)], ignore_index=False)
    #         show_plots(df)

    #         # write the output frame to disk, repeat (time taken * 30 fps) in order to get a video at real speed
    #     if save_video:
    #         frame_cnt = int((time.time()-start)*fps) if webcam and realspeed else 1
    #         save_frame(writer, frame, frame_cnt)

    #     end = time.time()
    #     print("[INFO] Total handling  : %2.1f sec" % (end - start))
    #     print("[INFO] People in frame : {}".format(npeople))
    #     if print_ascii:
    #         print_ascii_large(str(npeople)+ (" persons" if npeople > 1 else " person"))
    #     # Check for exit
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #     fps.update()
    # # release the file pointers
    # print("[INFO] cleaning up...")
    # fps.stop()
    # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # if save_video:
    #     writer.release()
    # cam.release()
  
