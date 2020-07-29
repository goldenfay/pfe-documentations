import ctypes
import signal
from sockets import ClientSocket
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from matplotlib import cm
from io import BytesIO as _BytesIO
from PIL import Image
import re,time,base64,os,sys,glob,datetime,traceback,inspect
from flask import Flask, Response
from werkzeug.serving import make_server
import dill
import multiprocessing
from multiprocessing import Pool
import threading
from threading import Thread


currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
sys.path.append(currentdir)
# User's modules
from modelmanager import ModelManager

#from sockets import ClientSocket
from components.base import Component
import components.reusable as reusable
import components.static as static
import functions


from app import app,get_regions_params

class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self,  *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class ServerThread(threading.Thread):

    def __init__(self, srv: Flask):
        threading.Thread.__init__(self)
        self.srv = make_server('127.0.0.1', 4000, srv)
        # self.srv = srv
        self._stopper = threading.Event()
        self.ctx = srv.app_context()
        # self.ctx.push()

    def run(self):
        print('starting process server')

        try:
            # self.srv.run('127.0.0.1', 4000)
            self.srv.serve_forever()
        except:
            self._stopper.set()

        # self.srv.serve_forever()

    def shutdown(self):
        self.srv.shutdown()
        # print('shutted down')
        self._stopper.set()
        # self._stop()
        print('[Server] Terminated')

    def stopped(self):
        return self._stopper.is_set()

    def get_id(self):

            # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
                                                         ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')


images_list = []
res_img_list = []

HTML_IMG_SRC_PREFIX = 'data:image/png;base64, '
config = None
ONLINE_MODE = False
server = None
server_thread = None
SERVER_URL = ''
CLIENT_SOCKET = None
best_performence_models = {
    'MCNN': 'internal',
    'CSRNet': 'external',
    'SANet': 'external',
    'CCNN': 'internal'
}


def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)


def apply_async(pool, fun, args):
    payload = dill.dumps((fun, args))
    return pool.apply_async(run_dill_encoded, (payload,))


def parse_contents(contents, filename):
    global images_list
    images_list.append(contents.encode("utf-8").split(b";base64,")[1])
    return html.Div(children=[
        html.H5(filename, style={'textAlign': 'center'}),


        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents, id=re.sub('\.d+', '', filename), style={
            'maxWidth': '80%', 'minWidth': '80px'})

    ],
        className='col-md justify-content-center animate__animated animate__fadeInRight display-upload')


class View(Component):
    layout = None

    def __init__(self, app, config):
        self.config = config
        super(View, self).__init__(app)

    def initialize(self, app):
        global images_list,res_img_list
        images_list = []
        res_img_list=[]
        list_vidoes = list(glob.glob(os.path.join(
            self.config.VIDEOS_DIR_PATH, '*.mp4')))

        # Main Layout
        self.layout = html.Div(
            children=[
                html.Div(
                    id='top-bar',
                    className='row',
                    style={'backgroundColor': '#fa4f56',
                           'height': '5px',
                           }
                ),
                html.Div(
                    className='row',
                    children=[
                        html.Div(
                            id='left-side-column',
                            className='col-md-8 d-flex flex-column align-items-center',
                            style={

                                'backgroundColor': '#F2F2F2'
                            },
                            children=[
                                static.default_header(),
                                html.Div(
                                    id='footage-container',
                                    children=static.default_footage_section(),
                                    className='col-md-12 mt-5'


                                )


                            ]
                        ),
                        html.Div(
                            id='right-side-column',
                            className='col-md-4',
                            style={

                                'overflowY': 'scroll',

                                'backgroundColor': '#F9F9F9'
                            },
                            children=[
                                html.Div(
                                    className='control-section d-flex flex-column',
                                    children=[
                                        html.Div(
                                            className='row',
                                            children=[
                                                html.Div(
                                                    className='col-md-12 d-flex justify-content-center',
                                                    children=[
                                                        html.H4(
                                                            'General Settings', className='muted')
                                                    ]
                                                )
                                            ]
                                        ),
                                        html.Div(
                                            className='control-element',
                                            children=[
                                                html.Div(
                                                    children=['Usage:'],
                                                    style={
                                                        'width': '40%'}
                                                ),
                                                html.Div(
                                                    children=[daq.ToggleSwitch(
                                                        id='usage-switch',
                                                        value=False,
                                                        color='#fa4f56'
                                                    )
                                                    ],
                                                    style={
                                                        'width': '20%'}
                                                ),

                                                html.Div(children=['Local'],
                                                         id='usage-switch-label',
                                                         style={
                                                    'width': '40%',
                                                    'textAlign': 'center'}
                                                )
                                            ]
                                        ),
                                        html.Div(
                                            id="server-url-control",
                                            className='control-element',
                                            children=[
                                                html.Div(
                                                    children=["Server URL:"], style={'width': '40%'}),
                                                html.Div(dcc.Input(
                                                    id="server-url-input",
                                                    type="text",
                                                    placeholder="Server Url",
                                                    style={
                                                        'display': 'inline-block'},
                                                ), style={'width': '60%'})
                                            ]
                                        ),
                                        html.Div(
                                            className='row',
                                            children=[
                                                html.Div(
                                                    className='col-md-12 d-flex justify-content-center',
                                                    children=[
                                                        html.H4(
                                                            'advanced Settings', className='muted')
                                                    ]
                                                )
                                            ]
                                        ),
                                        html.Div(
                                            className='control-element',
                                            children=[
                                                html.Div(
                                                    children=['Mode:'],
                                                    style={
                                                        'width': '40%'}
                                                ),
                                                html.Div(
                                                    children=[daq.ToggleSwitch(
                                                        id='mode-switch',
                                                        value=True,
                                                        color='#fa4f56'
                                                    )
                                                    ],
                                                    style={
                                                        'width': '20%'}
                                                ),

                                                html.Div(children=['Footage'],
                                                         id='switch-label',
                                                         style={
                                                    'width': '40%',
                                                    'text1lign': 'center'}
                                                )
                                            ]
                                        ),
                                        html.Div(style={'display': 'none'},
                                                 className='control-element',
                                                 children=[
                                            html.Div(
                                                children=["FPS (Frames/second):"], style={'width': '40%'}),
                                            html.Div(dcc.Slider(
                                                id='slider-fps',
                                                min=10,
                                                max=70,
                                                marks={
                                                    i: f'{i}' for i in range(10, 71, 10)},
                                                value=20,
                                                updatemode='drag'
                                            ), style={'width': '60%'})
                                        ]
                                        ),
                                        reusable.dropdown_control("Model type:", [
                                            {'label': 'Detection models:',
                                             'value': 'DM', 'disabled': True},
                                            {'label': 'Mobile SSD',
                                             'value': 'mobileSSD'},
                                            {'label': 'YOLO',
                                             'value': 'yolo'},
                                            {'label': 'Density map based models:',
                                             'value': 'CNCC', 'disabled': True},
                                            {'label': 'MCNN',
                                             'value': 'MCNN'},
                                            {'label': 'CSRNet',
                                             'value': 'CSRNet'},
                                            {'label': 'SANet',
                                             'value': 'SANet'},
                                            {'label': 'CCNN',
                                             'value': 'CCNN'}
                                        ], "mobileSSD",
                                            id="dropdown-model-selection"

                                        ),
                                        html.Div(children=reusable.dropdown_control("Footage Selection:", [
                                            {'label': os.path.basename(
                                                el), 'value': el}
                                            for el in list_vidoes
                                        ],
                                            list_vidoes[0],
                                            id="dropdown-footage-selection"


                                        ),
                                            id='footage-selection-control'
                                        ),
                                        reusable.dropdown_control("Video Display Mode:", [
                                            {'label': 'Normal Display',
                                             'value': 'normal'},
                                            {'label': 'Display with density map',
                                             'value': 'density_map'},
                                        ], 'density_map', id="dropdown-video-display-mode"
                                        ),

                                        reusable.dropdown_control("Graph View Mode:", [
                                            {'label': 'Visual Mode',
                                             'value': 'visual'},
                                            {'label': 'Detection Mode',
                                             'value': 'detection'}
                                        ], 'visual',

                                            id="dropdown-graph-view-mode",

                                        ),
                                        html.Div(id="div-visual-mode"),
                                        html.Div(id="div-detection-mode"),
                                        html.Div(
                                            id="socket-errors-div", children=[]),
                                        html.Div(id="test-div", children=[])
                                    ]
                                )]),
                        static.markdown_popup(),
                        html.Div(id='dummy-div', style={'display': 'none'})
                    ]
                )
            ]
        )


##############################################################################################
#           Dropdowns event handlers
##############################################################################################

# Switching Usage mode (local/online)

@app.callback([Output("usage-switch-label", "children"),
               Output("server-url-control", "style"),
               Output("usage-switch", "className")],
              [Input("usage-switch", "value")],
              [State("usage-switch", "className")])
def toggle_usage(value, classname):
    global ONLINE_MODE
    ONLINE_MODE = value
    if classname is None:
        classname = ''
    if not value:  # Local usage
        return ['Local'], {'display': 'none'}, classname.replace('toggled-on', '')
    else:  # online usage
        return ['Online'], {'display': 'block'}, classname+'toggled-on'

# Handle server-url input


@app.callback(
    Output("server-url-control", "children"),
    [Input("server-url-input", "value")],
    [State("server-url-control", "children")]
)
def server_url_change(value, children):
    global SERVER_URL
    if value != '':
        SERVER_URL = value
    return children


# Switching layouts


@app.callback([Output("switch-label", "children"),
               Output('footage-container', 'children'),
               Output('footage-selection-control', 'style')],
              [Input("mode-switch", "value")])
def toggle_display(value):
    if value:
        children = [
            html.Div(
                className='md-12 d-flex align-items-center flex-column',
                children=[
                    reusable.drag_drop_container('upload-image', 'drop-div', ['Drag and Drop', 'or', 'Select Files']
                                                 ),
                    html.Div(children=[html.Div(id='output-image-upload',
                                className='row d-flex align-items-center justify-content-center flex-row')]),
                    html.Div(children=[html.Div(id='output-image-process',
                     className='d-flex flex-column', children=[''])])
                ])
        ]
    else:
        children = [
            html.Div(
                className='md-12 d-flex align-items-center flex-column',
                children=[
                    html.Div(
                        id='',
                        className='d-flex align-items-center justify-content-center',
                        children=[
                            html.Button(
                                id='process-video-button',
                                n_clicks=0,
                                className='btn btn-lg btn-outline-success',
                                children=[html.Span('Start processing video', className='mr-2'), html.Span(
                                    className='fa fa-play')],
                                style={'fontWeight': 'bold',
                                       'fontSize': '26px',
                                       'minWidth': '100px',
                                       'minHeight': '60px'}

                            )
                        ],
                        style={
                            'width': '500px',
                            'height': '300px',
                        }
                    ),
                    html.Div(id='output-video-process',
                                className='container')
                ])
        ]
        # static.default_footage_section()
    return ['Footage' if not value else 'Still images'], children, {'display': 'block'} if not value else {'display': 'none'}

    # Footage Selection
@app.callback(Output("video-display", "url"),
              [Input('dropdown-footage-selection', 'value')])
def select_footage(footage):
    # Find desired footage and update player video
    # url = url_dict[footage]
    return 'url'

    # Model selection
@app.callback(Output("dummy-div", "style"),
              [Input("dropdown-model-selection", "value")])
def change_model(model_type):
    if ONLINE_MODE:
        return {"display": "none"}
    print('[INFO] Loading model : ', model_type, ' ...')
    if model_type in ['mobileSSD', 'yolo']:
        x = ModelManager.load_detection_model(model_type)
        print(type(x))
    else:
        external_flag = best_performence_models[model_type] == 'external'
        try:
            ModelManager.load_external_model(model_type, external_flag)
        except Exception as e:
            print('An error occured when loading model ',
                  model_type, end='\n\t')
            traceback.print_exc()
            pass
    print('[INFO] Done.')
    return {"display": "none"}

    # Upload image
@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'filename'), Input('upload-image', 'contents')])
def update_output(list_of_names, list_of_contents):
    global ONLINE_MODE,images_list
    images_list=[]
    if list_of_contents is not None:
        print('[INFO] uploading...')
        children = [html.H3('Preview',className='ml-5 text-primary font-weight-bold flex-break'),html.Hr()]
        for i in range(len(list_of_contents)):
            children.append(parse_contents(
                list_of_contents[i], list_of_names[i]))
        children += [html.Div(children=[html.Button(
            id='process-imgs-button',
            n_clicks=0,
            className='btn btn-md btn-outline-success ' +
            ('socket-btn' if ONLINE_MODE else ''),
            children=[html.Span(className='fa fa-play')])],
            style={'fontWeight': 'bold',
                   'fontSize': '20px',
                   'minWidth': '40px',
                   'minHeight': '40px'}
        ),
            
        ]

        print('[INFO] Done')
        return children

        # Upload video


@app.callback(
    Output('output-video-upload', 'children'),
    [Input("upload-video", "filename"), Input("upload-video", "contents")],
)
def update_output(uploaded_filenames, uploaded_file_contents):
        # if uploaded_filenames is not None and uploaded_file_contents is not None:
    var = ""
    if uploaded_filenames is not None:
        var = uploaded_filenames[0]
        return html.Div(html.Video(id="myvideo", controls=True, src='http://127.0.0.1:8080/'+var, style={'width': '500px', 'height': '300px'}))


##############################################################################################
#           Buttons clicks event handlers
##############################################################################################

    # Start detection click


@app.callback([Output("output-image-process", "children"),
               Output("socket-errors-div", "children")],
              [Input("process-imgs-button", "n_clicks")],
              [State("dropdown-model-selection", "value"),
               State("output-image-process", "children")])
def launch_counting(button_click, model_type, children):
    global images_list, res_img_list, CLIENT_SOCKET
    res_img_list=[]
    # return [],[] # to be removed
    if button_click == 0:
        return [], []
    else:
            # Convert html images (base64 encoded) to numpy arrays
        frames = [functions.b64_to_numpy(el) for el in images_list]

        # results=ModelManager.process_frame(frames)
        # res_img_list=[]
        # for img,count in enumerate(results):
        #     print('Count : ', count)
        #     encoded_img = numpy_to_b64(img,model_type not in ['mobileSSD','yolo'])
        #     res_img_list.append(encoded_img)

        if ONLINE_MODE:  # Proceed images to server and wait for results
            received = False
            server_error = False

            def response_received(data):
                print('Process is done in server, message received !')
                received = True

            def server_error_response(data):
                error_msg = data['message']
                server_error = True
            if CLIENT_SOCKET is None:
                CLIENT_SOCKET = ClientSocket(reconnection=False)
                CLIENT_SOCKET.on('server-error', handler=server_error_response)
                CLIENT_SOCKET.on('send-image', handler=append_res_img)
                CLIENT_SOCKET.on('process-done', handler=response_received)
            if not CLIENT_SOCKET.connected:
                CLIENT_SOCKET.connect(SERVER_URL)

            images = [{
                'id': 'img'+str(idx),
                'index': idx,
                'data': images_list[idx]
            }
                for idx, frame in enumerate(frames)
            ]
            data = {
                'model_type': model_type,
                'images': images
            }
            print('[INFO] Sending images to server ...')
            # CLIENT_SOCKET.emit('image-upload', data)
            print('[INFO] Done.')
            # while True:
            #     if received or server_error:
            #         break
            #     print('waiting ...')
            #     time.sleep(1)
            if server_error:
                return [], [
                    dbc.Alert(children=['An error occured on the server.'],
                              n_clicks=0,
                              color="danger",
                              id="socket-error-title"
                              ),
                    dbc.Collapse(children=[
                        dbc.Card(dbc.CardBody(error_msg))],
                        id="collapse",
                    ),

                ]
            return [],[]    

        else:
            for id, frame in enumerate(frames):
                print('Processing image :\n\t frame of shape : ', frame.shape)
                try:

                    start = time.time()
                    dmap, count = ModelManager.process_frame(frame)
                    inference_time = time.time()-start

                    print('\t Inference time : ',
                          inference_time, 'count : ', count)
                    encoded_img = functions.numpy_to_b64(
                        dmap, model_type not in ['mobileSSD', 'yolo'])
                    res_img_list.append(
                        (id, HTML_IMG_SRC_PREFIX+encoded_img, count))
                except Exception as e:
                    print("An error occured while detecting ", end='\n\t')
                    traceback.print_exc()

            return [html.H3('Output',className='ml-5 text-primary font-weight-bold flex-break'),html.Hr()]\
                +reusable.count_results_grid([
                HTML_IMG_SRC_PREFIX+(el.decode("utf-8"))
                for el in images_list
            ], res_img_list), []

    # Process video button click


@app.callback(Output("output-video-process", "children"),
              [Input("process-video-button", "n_clicks")],
              [State("dropdown-model-selection", "value"),
               State("dropdown-footage-selection", "value")])
def process_video(button_click, model_type, video_path):
    global server, server_thread, app, get_regions_params,SERVER_URL
    if button_click > 0:
        print('Process server state : ',('not None' if server is not None else 'None'))
        if server_thread is not None:
            print('\t Thread alive ?: ',server_thread.isAlive())

        # def launch_subprocess(model_type, video_path, queue):
        #     global server
        #     from flask import request

        #     if model_type in ['mobileSSD', 'yolo']:
        #         x = ModelManager.load_detection_model(model_type)
        #         print('\t', type(x))
        #     else:
        #         try:
        #             ModelManager.load_external_model(model_type)
        #         except Exception as e:
        #             print('An error occured when loading model ',
        #                   model_type, end='\n\t')
        #             traceback.print_exc()
        #             pass
        #     print('[INFO] Done.')
        #     # ModelManager.process_video(video_path,queue)

        #     if server is None:
        #         server = Flask(__name__)
        #         server.config['PRESERVE_CONTEXT_ON_EXCEPTION'] = False

        #         @server.route('/stream')
        #         def video_feed():
        #             return Response(ModelManager.process_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


        #         @server.route('/shutdown', methods=['GET'])
        #         def shutdown():

        #             func = request.environ.get('werkzeug.server.shutdown')
        #             if func is None:
        #                 raise RuntimeError(
        #                     'Not running with the Werkzeug Server')
        #             func()
        #             return 'Server shutting down...'
        #     server.run(port=4000)

        if model_type in ['mobileSSD', 'yolo']:
            x = ModelManager.load_detection_model(model_type)
            print('\t', type(x))
        else:
            try:
                ModelManager.load_external_model(model_type)
            except Exception as e:
                print('An error occured when loading model ',
                      model_type, end='\n\t')
                traceback.print_exc()
                pass
        print('[INFO] Done.')

        if ONLINE_MODE:
            if CLIENT_SOCKET is None:
                CLIENT_SOCKET = ClientSocket(reconnection=False)
                CLIENT_SOCKET.on('server-error', handler=server_error_response)
                CLIENT_SOCKET.on('send-frame', handler=append_res_img)
            if not CLIENT_SOCKET.connected:
                CLIENT_SOCKET.connect(SERVER_URL)

            CLIENT_SOCKET.emit('init-process-video',{'model_type':model_type})
            vs = cv2.VideoCapture(video_path)
            while True:
                frame = vs.read()
                    #VideoStream returns a frame, VideoCapture returns a tuple
                frame = frame[1] if len(frame)>1 else frame
                if frame is None:
                    break
                frame = imutils.resize(frame, width=500)
                CLIENT_SOCKET.emit('process-frame',{'frame':frame})
                key = cv2.waitKey(10) & 0xFF

                if key == ord("q"):
                    break

        else:
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
                
                    params={
                        'show':True,
                        'tang': float(get_regions_params()['tang']),
                        'b': int(float(get_regions_params()['b']))
                    } if get_regions_params() is not None else None
                    return Response(ModelManager.process_video(video_path,args={'regions_params':params}), mimetype='multipart/x-mixed-replace; boundary=frame')

            # server_thread=StoppableThread(target=launch_subprocess,args=(model_type,video_path,None))
            # server_thread.start()
            server_thread = ServerThread(server)
            # server_thread.setDaemon(True)
            server_thread.start()

            # pool = Pool(2)
            # job=apply_async(pool,launch_subprocess,(model_type,video_path,QUEUE))

            # ModelManager.process_video(video_path)

            # run=lambda :server.run(port=4000)
            # server_thread=multiprocessing.Process(target=server.run)
            # server_thread.start()

        return [
            html.Div(
                className='row shadow-sm',
                children=[
                    html.Div(
                        className='col-md-12 d-flex justify-content-center align-items-center',
                        id='edit-canvas-panel',
                        children=[
                            html.Button(id='line-canvas-button', children=[html.Span(
                                className='fa fa-pencil-alt')], className='btn mr-5', title='Draw a vertical line'),
                            html.Button(children=[html.Span(
                                className='fa fa-times')], className='btn ml-5', id='clear-canvas-button', title='Cancel')

                        ]
                    )
                ]

            ),
            html.Div(
                className="row mt-5 mb-5",
                children=[
                    html.Div(
                        className='col-md-12 d-flex justify-content-center align-items-center',
                        children=[
                            # html.Img(src='/video_feed')
                            html.Img(src='http://localhost:4000/stream?t='+str(datetime.datetime.now()),
                                     id='process-video-output-flow',
                                     className='img-fluid'),
                            # html.Iframe(src='http://localhost:4000/stream')
                        ]
                    )


                ]
            ),

            html.Div(
                id='edit-canvas-area',
                className='row d-none mt-5',
                children=[
                    html.Hr(
                        className='divider silent mt-5 mb-5'
                    ),
                    html.Div(
                        className='col-md-12 mt-5 d-flex flex-column justify-content-center',
                        children=[
                            html.H3('Scene Edition',className='text-primary text-center'),
                            html.P('Draw the regions you want to focus on',className='text-secondary text-center')

                        ]
                    ),
                    html.Div(
                        className='col-md-12 mt-5 d-flex justify-content-center align-items-center',
                        children=[html.Canvas(id='scene-canvas')]
                    ),
                    html.Div(
                        className='col-md-12 mt-5 shadow-sm d-flex justify-content-center align-items-center',
                        children=[
                            html.Button(id='confirm-draw-btn', className='btn-success', children=[
                                        html.Span(className='fa fa-check')], title='Confirm zones plit')
                        ]
                    )
                ]
            ),
            html.P(id='hidden-splitLine-input',
                      className='d-none', title='')




        ]





@app.callback(
    Output("hidden-splitLine-input", "className"),
    [Input("confirm-draw-btn", "n_clicks")]
)
def setup_splitlines(n_clicks):
    global server, server_thread
    
    if n_clicks is not None and n_clicks > 0:
        # server_thread.shutdown()
        import requests
        server_thread.raise_exception()
        # server_thread.join()
        # requests.get('http://localhost:4000/terminate')
        server_thread.shutdown()
        # server_thread.join()
        print(server_thread.isAlive())
        # server_thread.terminate()
        # server_thread.stop()
        server = None
    return 'd-none'



    # Sockets errors collapsing handler
@app.callback(
    Output("collapse", "is_open"),
    [Input("socket-error-title", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

    # Learn more popup
@app.callback(Output("markdown", "style"),
              [Input("learn-more-button", "n_clicks"), Input("markdown_close", "n_clicks")])
def update_click_output(button_click, close_click):
    if button_click > close_click:
        return {"display": "block"}
    else:
        return {"display": "none"}


if False:
    pass
    # Updating Figures

    # @app.callback(Output("count-evolution-graph", "figure"),
    #                 [Input("interval-detection-mode", "n_intervals")],
    #                 [State("video-display", "currentTime"),
    #                 State('dropdown-footage-selection', 'value'),
    #                 State('slider-fps', 'value')])
    # def update_score_bar(n, current_time, footage, threshold):
    #     layout = go.Layout(
    #         showlegend=False,
    #         paper_bgcolor='rgb(249,249,249)',
    #         plot_bgcolor='rgb(249,249,249)',
    #         xaxis={
    #             'automargin': True,
    #         },
    #         yaxis={
    #             'title': 'Score',
    #             'automargin': True,
    #             'range': [0, 1]
    #         }
    #     )

    #     if current_time is not None:
    #         current_frame = round(current_time * config.FRAMERATE)

    #         if n > 0 and current_frame > 0:
    #             pass

    #             figure = go.Figure({
    #                 'data': [{'hoverinfo': 'x+text',
    #                           'name': 'Detection Scores',
    #                           'text': y_text,
    #                           'type': 'bar',
    #                           'x': objects_wc,
    #                           'marker': {'color': colors},
    #                           'y': frame_df["score"].tolist()}],
    #                 'layout': {'showlegend': False,
    #                            'autosize': False,
    #                            'paper_bgcolor': 'rgb(249,249,249)',
    #                            'plot_bgcolor': 'rgb(249,249,249)',
    #                            'xaxis': {'automargin': True, 'tickangle': -45},
    #                            'yaxis': {'automargin': True, 'range': [0, 1], 'title': {'text': 'Score'}}}
    #                 }
    #             )
    #             return figure

    #     # Returns empty bar
    #     return go.Figure(data=[go.Bar()], layout=layout)

    # @app.callback(Output("pie-object-count", "figure"),
    #               [Input("interval-visual-mode", "n_intervals")],
    #               [State("video-display", "currentTime"),
    #                State('dropdown-footage-selection', 'value'),
    #                State('slider-fps', 'value')])
    # def update_object_count_pie(n, current_time, footage, threshold):
    #     layout = go.Layout(
    #         showlegend=True,
    #         paper_bgcolor='rgb(249,249,249)',
    #         plot_bgcolor='rgb(249,249,249)',
    #         autosize=False,
    #         margin=go.layout.Margin(
    #             l=10,
    #             r=10,
    #             t=15,
    #             b=15
    #         )
    #     )

    #     if current_time is not None:
    #         current_frame = round(current_time * FRAMERATE)

    #         if n > 0 and current_frame > 0:
    #             video_info_df = data_dict[footage]["video_info_df"]

    #             # Select the subset of the dataset that correspond to the current frame
    #             frame_df = video_info_df[video_info_df["frame"] == current_frame]

    #             # Select only the frames above the threshold
    #             threshold_dec = threshold / 100  # Threshold in decimal
    #             frame_df = frame_df[frame_df["score"] > threshold_dec]

    #             # Get the count of each object class
    #             class_counts = frame_df["class_str"].value_counts()

    #             classes = class_counts.index.tolist()  # List of each class
    #             counts = class_counts.tolist()  # List of each count

    #             text = [f"{count} detected" for count in counts]

    #             # Set colorscale to piechart
    #             colorscale = ['#fa4f56', '#fe6767', '#ff7c79', '#ff908b', '#ffa39d', '#ffb6b0', '#ffc8c3', '#ffdbd7',
    #                           '#ffedeb', '#ffffff']

    #             pie = go.Pie(
    #                 labels=classes,
    #                 values=counts,
    #                 text=text,
    #                 hoverinfo="text+percent",
    #                 textinfo="label+percent",
    #                 marker={'colors': colorscale[:len(classes)]}
    #             )
    #             return go.Figure(data=[pie], layout=layout)

    #     return go.Figure(data=[go.Pie()], layout=layout)  # Returns empty pie chart

##############################################################################################################
##############################################################################################################
############################################SOCKETS HANDLERS #################################################
##############################################################################################################
##############################################################################################################


def append_res_img(data):
    global res_img_list
    print('[INFO] A result received from server.')
    res_img_list.append(
        (data['id'], data['data'], data['count']))


def receive_result_image(data):
    return html.Div(className='row', children=[

        html.Div(children=[
                    html.Div(html.H4('Original', className='muted'),
                             className="d-flex justify-content-center"),
                    html.Img(id='img-org-{}'.format(id),
                             src=HTML_IMG_SRC_PREFIX+(images_list[i].decode("utf-8")), style={
                        'width': '100%'})

                    ],
                 className='col-md justify-content-center animate__animated animate__fadeInRight'),
        html.Div(children=[
            html.Div(html.H4('Estimated count : '+str(int((count+1)/100)),
                             className='muted'), className="d-flex justify-content-center"),
            html.Img(id='img-{}'.format(id),
                     src=encoded_img, style={
                        'width': '100%'})

        ],
            className='col-md justify-content-center animate__animated animate__fadeInRight')
    ])
