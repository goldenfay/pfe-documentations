from flask import Flask,Response
from engineio.payload import Payload
from flask_socketio import SocketIO, emit, join_room, leave_room
import sys,os,glob,inspect,time,traceback,json,base64
import numpy as np
import torch
import cv2 

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(currentdir)
    # Custom modules
import server_config
from thread_server import ServerThread
sys.path.append(server_config.BASIC_TOOLS_ROOT_PATH)
import util.process_functions as process_functions
from modelmanager import ModelManager
import config
import imutils


ModelManager.set_base_path(config.FROZEN_MODELS_BASE_PATH)

    # Define flask app and flask-socketIo wrapper
app = Flask(__name__)

secret=os.getenv("SECRET_KEY")
app.logger.info("Starting...")
app.config['SECRET_KEY'] = secret
app.logger.critical("secret: %s" % secret)


Payload.max_decode_packets = 50000000
socketio = SocketIO(app,async_handlers=True,cors_allowed_origins="*",ping_timeout=600000,ping_interval=100)

HTML_IMG_SRC_PARAMETERS = 'data:image/png;base64, '
server=None
server_thread=None
model_type=None
list_frame=[]
writter=None
Ready=False
def load_model(model_type):
    try:
        if model_type in ['mobileSSD', 'yolo']:
            x = ModelManager.load_detection_model(model_type)
        else:
            
            ModelManager.load_external_model(model_type)
    except Exception as e:
        print('[Loading model] An error occured while loading model ',
                model_type, end='\n\t')
        traceback.print_exc()
        emit('server-error',{'message':str(e)})
        print('error sent')
        return
    print('Done.')

    #default routes
@app.route('/stream')
def video_feed():
    global Ready
    if not Ready:
        print('not ready yet')
        return 'Hello'
    else:   
        print('Ready to stream video')
        return Response(ModelManager.process_video('temp.avi'), mimetype='multipart/x-mixed-replace; boundary=frame')


    # Basic default socket event handlers
@socketio.on('connect')
def connected():
    print('Client connected')
    
@socketio.on('disconnect')
def disconnect():
    print('Client disconnected')

    # Count and process images event handlers

@socketio.on('image-upload')
def imageUpload(data):
    print('[image-upload] Socket data Received')
    model_type=data['model_type']
    images_list=data['images']
    print('[image-upload] Loading model ',model_type,' ...',end='\t')

    load_model(model_type)

    print('[image-upload] Converting images to arrays ...',end='\t')

    for image in images_list:
        image['data']=process_functions.b64_to_numpy(image['data'])
    print('Done.') 
    errors=[]
    print('[image-upload] Processing images ...')
    for id, frame in enumerate(images_list):
            print('\t Processing image :\n\t\t Id : ', frame['id'])
            try:

                start = time.time()
                res_img, count = ModelManager.process_frame(frame['data'])
                inference_time = time.time()-start
                print('\t Done. ')
                encoded_img = HTML_IMG_SRC_PARAMETERS+(process_functions.numpy_to_b64(
                    res_img, model_type not in ['mobileSSD', 'yolo']))
                data={
                    'id':frame['id'],
                    'index': frame['index'],
                    'data': encoded_img,
                    'count': count,
                    'time':str(inference_time)
                }    
                emit('send-image', data,broadcast = True)
            except Exception as e:
                print("An error occured while processing the image ", end='\n\t')
                traceback.print_exc()
                errors.append((frame['id'],str(e)))
                continue
    print('[image-upload] Processing is done'+(' with errors' if len(errors)>0 else ''),'.')  
    emit('process-done',{'flag': 'success' if len(errors)==0 else 'fail','errors':errors},broadcast = True)         


@socketio.on('init-process-video')
def setup(data):
    global model_type,writter
    model_type=data['model_type']
    H,W=data['height'],data['width']
    print('[init-process-video] Loading model ',model_type,' ...',end='\t')
    load_model(model_type)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writter = cv2.VideoWriter('temp.avi', fourcc, 30,
        (W, H), True)

@socketio.on('process-frame')
def imageUpload(data):
    global model_type
    print('[process-frame] frame Received')
    if ModelManager.model is None:
        print('[process-frame] Loading model ',model_type,' ...',end='\t')

        load_model(model_type)

    print('[process-frame] Converting frame  ...',end='\t')

    
    frame=process_functions.b64_to_numpy(data['frame'])
    print('Done.') 
    errors=[]
    print('[process-frame] Processing frame ...')
    
    try:

        start = time.time()
        res_img, count = ModelManager.process_frame(frame)
        inference_time = time.time()-start
        
        
        print('\t Done. ')
        if model_type not in ['mobileSSD', 'yolo']:
            concatenated=process_functions.concat_frame_dmap(frame,res_img)
            # while True:
            #     cv2.imshow('hh',frame)
            #     cv2.imshow('img',concatenated)
            #     key = cv2.waitKey(10) & 0xFF

            #     if key == ord("q"):
            #         break
            # concatenated=cv2.cvtColor(concatenated, cv2.COLOR_BGR2RGB)
            # encoded_img = HTML_IMG_SRC_PARAMETERS+(process_functions.numpy_to_b64(
            # concatenated, False))
            cnt = cv2.imencode('.png',concatenated)[1]
            encoded_img = HTML_IMG_SRC_PARAMETERS+base64.b64encode(cnt).decode('utf-8')
        else:
            cnt = cv2.imencode('.png',res_img)[1]
            encoded_img = HTML_IMG_SRC_PARAMETERS+base64.b64encode(cnt).decode('utf-8')
        data={
            'data': encoded_img,
            'count': count,
            'time':str(inference_time)
        }    
        emit('send-frame', data,broadcast = True)
    except Exception as e:
        print("An error occured while processing the frame ", end='\n\t')
        traceback.print_exc()
        emit('process-frame-error',{},broadcast = True)         
        
    print('[process-frame] Processing is done'+(' with errors' if len(errors)>0 else ''),'.')  

   


@socketio.on('frame-upload')
def frameUpload(data):
    global list_frame,writter
    list_frame.append(data['frame'])
    frame=process_functions.b64_to_numpy(data['frame'])
   
    writter.write(frame)
    print('Frame received')

@socketio.on('process-video')
def startprocessing(data):
    global list_frame,model_type,writter,Ready
    writter.release()
    Ready=True
    print('Video received.')




    

    # for frame in list_frame:
    #     try:

    #             start = time.time()
    #             res_img, count = ModelManager.process_frame(frame)
    #             inference_time = time.time()-start

    #             print('\t Done. ')
    #             encoded_img = HTML_IMG_SRC_PARAMETERS+(process_functions.numpy_to_b64(
    #                 res_img, model_type not in ['mobileSSD', 'yolo']))
    #             data={
    #                 'data': encoded_img,
    #                 'count': count,
    #                 'time':str(inference_time)
    #             }    
    #             emit('send-frame', data,broadcast = True)
    #     except Exception as e:
    #         print("An error occured while processing the image ", end='\n\t')
    #         traceback.print_exc()
    #         errors.append((frame['id'],str(e)))
    #         continue
    # print('[process-video] Processing is done'+(' with errors' if len(errors)>0 else ''),'.')  
    # # emit('process-done',{'flag': 'success' if len(errors)==0 else 'fail','errors':errors},broadcast = True)         



@socketio.on('video-upload')
def imageUpload(data):
    global server,server_thread
    print(type(data))
    emit('send-image', 'video uploaded')

    if server is None:
            print('[SERVER] Creating a server instance ...')
            from flask import request
            server = Flask('StreamServer')

            @server.route('/test', methods=['GET'])
            def test():
                return 'jfhskdjfhskjdhfkjshdkjfhskdjhf'

            @server.route('/terminate', methods=['GET'])
            def stop_server():
                raise threading.ThreadError("the thread is not active")

            @server.route('/stream')
            def video_feed():
               
                return Response(ModelManager.process_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

        
    server_thread = ServerThread(server)
    server_thread.start()

    
        
if __name__ == '__main__':
    # config.print_ascii_large('Crowd Countinf Process Server',font_size=7)
    print('''
 ██████ ██████   ██████  ██     ██ ██████       ██████  ██████  ██    ██ ███    ██ ████████ ██ ███    ██  ██████  
██      ██   ██ ██    ██ ██     ██ ██   ██     ██      ██    ██ ██    ██ ████   ██    ██    ██ ████   ██ ██       
██      ██████  ██    ██ ██  █  ██ ██   ██     ██      ██    ██ ██    ██ ██ ██  ██    ██    ██ ██ ██  ██ ██   ███ 
██      ██   ██ ██    ██ ██ ███ ██ ██   ██     ██      ██    ██ ██    ██ ██  ██ ██    ██    ██ ██  ██ ██ ██    ██ 
 ██████ ██   ██  ██████   ███ ███  ██████       ██████  ██████   ██████  ██   ████    ██    ██ ██   ████  ██████  
                                                                                                                  
                                                                                                                  
██████  ██████   ██████   ██████ ███████ ███████ ███████     ███████ ███████ ██████  ██    ██ ███████ ██████      
██   ██ ██   ██ ██    ██ ██      ██      ██      ██          ██      ██      ██   ██ ██    ██ ██      ██   ██     
██████  ██████  ██    ██ ██      █████   ███████ ███████     ███████ █████   ██████  ██    ██ █████   ██████      
██      ██   ██ ██    ██ ██      ██           ██      ██          ██ ██      ██   ██  ██  ██  ██      ██   ██     
██      ██   ██  ██████   ██████ ███████ ███████ ███████     ███████ ███████ ██   ██   ████   ███████ ██   ██     
                                                                                                                  
                                                                                                                  
''')
    socketio.run(app,port=5000, debug=True)
    