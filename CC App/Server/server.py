from flask import Flask
from flask_socketio import SocketIO, emit, join_room, leave_room
import sys,os,glob,inspect,time,traceback
import numpy as np
import torch

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(currentdir)
import server_config
sys.path.append(server_config.BASIC_TOOLS_ROOT_PATH)
import util.process_functions as process_functions
from modelmanager import ModelManager
import config


    # Define flask app and flask-socketIo wrapper
app = Flask(__name__)

secret=os.getenv("SECRET_KEY")
app.logger.info("Starting...")
app.config['SECRET_KEY'] = secret
app.logger.critical("secret: %s" % secret)
socketio = SocketIO(app,cors_allowed_origins="*",ping_timeout=600000)

HTML_IMG_SRC_PARAMETERS = 'data:image/png;base64, '

    #default routes
@app.route('/hello')
def hello():
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
    print('[image-upload] Loading model ',model_type,' ...')

    if model_type in ['mobileSSD', 'yolo']:
        x = ModelManager.load_detection_model(model_type)
    else:
        try:
            ModelManager.load_external_model(model_type)
        except Exception as e:
            print('[image-upload] An error occured when loading model ',
                  model_type, end='\n\t')
            traceback.print_exc()
            pass
    print('Done.')
    print('[image-upload] Converting images to arrays ...')

    for image in images_list:
        image['data']=process_functions.b64_to_numpy(image['data'].encode("utf-8").split(b";base64,")[1])
    print(images_list[:2])    
    errors=[]
    print('[image-upload] Processing images ...')
    for id, frame in enumerate(images_list):
            print('\t Processing image :\n\t\t Id : ', frame['id'])
            try:

                start = time.time()
                res_img, count = ModelManager.process_frame(frame)
                inference_time = time.time()-start

                print('\t Done. ')
                encoded_img = HTML_IMG_SRC_PARAMETERS+process_functions.numpy_to_b64(
                    res_img, model_type not in ['mobileSSD', 'yolo'])
                data={
                    'id':frame['id'],
                    'index': frame['index'],
                    'data': encoded_img,
                    'time':inference_time
                }    
                emit('send-image', data, broadcast = True)
            except Exception as e:
                print("An error occured while detecting ", end='\n\t')
                traceback.print_exc()
                errors.append((frame['id'],list(sys.exc_info())))
                continue
    print('[image-upload] Processing is done'+(' with errors' if len(errors)>0 else ''),'.')  
    # emit('process-done',{'flag': 'success' if len(errors)==0 else 'fail','errors':errors},broadcast = True)         

    # result_img=process_functions.numpy_to_b64(np.zeros((250,250,3)),False)
    # print('sending result ...')
    # emit('send-image', result_img, broadcast = True)   



    
        
if __name__ == '__main__':
    socketio.run(app, debug=True)