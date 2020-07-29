from flask import Flask
from engineio.payload import Payload
from flask_socketio import SocketIO, emit, join_room, leave_room
import sys,os,glob,inspect,time,traceback,json
import numpy as np
import torch

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(currentdir)
import server_config
from thread_server import ServerThread
sys.path.append(server_config.BASIC_TOOLS_ROOT_PATH)
import util.process_functions as process_functions
from modelmanager import ModelManager
import config


ModelManager.set_base_path(config.FROZEN_MODELS_BASE_PATH)

    # Define flask app and flask-socketIo wrapper
app = Flask(__name__)

secret=os.getenv("SECRET_KEY")
app.logger.info("Starting...")
app.config['SECRET_KEY'] = secret
app.logger.critical("secret: %s" % secret)


Payload.max_decode_packets = 500
socketio = SocketIO(app,async_handlers=True,cors_allowed_origins="*",ping_timeout=600000,ping_interval=100)

HTML_IMG_SRC_PARAMETERS = 'data:image/png;base64, '
server=None
server_thread=None
model_type=None
list_frame=[]
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

@app.route('/hello')
def hello():
    global server,server_thread
    if server is None:
            print('[SERVER] Creating a server instance ...')
            from flask import request
            server = Flask('StreamServer')

            @server.route('/test', methods=['GET'])
            def test():
                return 'jfhskdjfhskjdhfkjshdkjfhskdjhf'
            PORT=4000
            server_thread = ServerThread(server,PORT)
            os.system('./ngrok http {} &'.format(PORT))
            os.system("curl  http://localhost:4040/api/tunnels > tunnels.json")
            with open('tunnels.json') as data_file:    
                datajson = json.load(data_file)
            listurls=[el['public_url'] for el in datajson['tunnels']]    
            print(listurls)
            server_thread.start()            
    return "Hello World!"

    # Basic default socket event handlers
@socketio.on('connect')
def connected():
    print('connect')
    
@socketio.on('disconnect')
def disconnect():
    print('disconnect')

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
    print('Getting ',len(images_list)) 
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
    global model_type
    model_type=data['model_type']
    print('[process-frame] Loading model ',model_type,' ...',end='\t')
    load_model(model_type)

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
        encoded_img = HTML_IMG_SRC_PARAMETERS+(process_functions.numpy_to_b64(
            res_img, model_type not in ['mobileSSD', 'yolo']))
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
    global list_frame
    list_frame.append(data['frame'])

@socketio.on('process-video')
def startprocessing(data):
    global list_frame,model_type
    for frame in list_frame:
        try:

                start = time.time()
                res_img, count = ModelManager.process_frame(frame)
                inference_time = time.time()-start

                print('\t Done. ')
                encoded_img = HTML_IMG_SRC_PARAMETERS+(process_functions.numpy_to_b64(
                    res_img, model_type not in ['mobileSSD', 'yolo']))
                data={
                    'data': encoded_img,
                    'count': count,
                    'time':str(inference_time)
                }    
                emit('send-frame', data,broadcast = True)
        except Exception as e:
            print("An error occured while processing the image ", end='\n\t')
            traceback.print_exc()
            errors.append((frame['id'],str(e)))
            continue
    print('[process-video] Processing is done'+(' with errors' if len(errors)>0 else ''),'.')  
    # emit('process-done',{'flag': 'success' if len(errors)==0 else 'fail','errors':errors},broadcast = True)         



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
    socketio.run(app,port=5000, debug=True)