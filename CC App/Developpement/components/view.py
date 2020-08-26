from sockets import ClientSocket
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_daq as daq
import dash_player as player
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import cv2,imutils
import re,time,base64,os,sys,glob,datetime,traceback,inspect
from flask import Flask, Response
import multiprocessing
from multiprocessing import Pool
import threading


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
from threads import *

from app import app,get_regions_params


images_list = []
res_img_list = []

HTML_IMG_SRC_PREFIX = 'data:image/png;base64, '
Lang = None
path_video=None
ONLINE_MODE = False
SHOW_GRAPHS=True
server = None
server_thread = None
SERVER_URL = ''
CLIENT_SOCKET = None
best_performence_models = {
    'MCNN': 'external',
    'CSRNet': 'external',
    'SANet': 'external',
    'CCNN': 'internal'
}



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
    config=None

    def __init__(self, app, config):
        View.config = config
        super(View, self).__init__(app)

    def reset_variables(self):
        global images_list,res_img_list,server,server_thread
        images_list = []
        res_img_list=[]
        if server_thread is not None and server_thread.isAlive():
            server_thread.raise_exception()
            

    def initialize(self, app):
        global Lang
        Lang=View.config.LANGUAGE_DICT
        static.Lang=Lang
        self.reset_variables()
        
        list_vidoes = list(glob.glob(os.path.join(
            View.config.VIDEOS_DIR_PATH, '*.mp4')))

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

                                'backgroundColor': '#FAFAFA'
                            },
                            children=[
                                html.Div(
                                    className='control-section d-flex flex-column',
                                    children=[
                                        html.Div(
                                            className='row mt-3',
                                            children=[
                                                html.Div(
                                                    className='col-md-12 d-flex justify-content-center',
                                                    children=[
                                                        html.H4(
                                                            Lang['General Settings'], className='muted')
                                                    ]
                                                )
                                            ]
                                        ),
                                        reusable.toggleswitch_control('Usage','usage-switch','usage-switch-label',False,Lang['Local'],'#fa4f56'),
                                        
                                        html.Div(
                                            id="server-url-control",
                                            className='control-element',
                                            children=[
                                                html.Div(
                                                    children=[Lang['Server URL:']], style={'width': '40%'}),
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
                                                            Lang['Advanced Settings'], className='muted')
                                                    ]
                                                )
                                            ]
                                        ),
                                        reusable.toggleswitch_control(Lang['Mode'],'mode-switch','switch-label',True,Lang['Footage'],'#fa4f56'),

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
                                        reusable.dropdown_control(Lang['Model type:'], static.model_selection_options(), "mobileSSD",
                                            id="dropdown-model-selection"

                                        ),
                                        html.Div(children=reusable.dropdown_control(Lang['Footage Selection:'], [
                                            {'label': os.path.basename(
                                                el), 'value': el}
                                            for el in list_vidoes
                                        ],
                                            list_vidoes[0] if len(list_vidoes)>0 else None,
                                            id="dropdown-footage-selection"


                                        ),
                                            id='footage-selection-control'
                                        ),

                                        html.Div(
                                            id='show-graphs-control',
                                            
                                            children=[
                                                reusable.toggleswitch_control(Lang['Show graphs:'],'show-graphs-switch','graph-switch-label',True,Lang['Yes'],'#fa4f56'),

                                                
                                            ]
                                        ),
                                        html.Div(id="display-plots-div",children=[]),
                                        static.default_count_plots_modal(os.path.join(ModelManager.outputs_path,'results_history.csv')),
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

def load_model(model_type):
    print('[INFO] Loading Model ',model_type,' ...')
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

    # graphs modal handler
@app.callback(
    Output("count-plots-modal", "is_open"),
    [Input("view-count-plots-btn", "n_clicks"), Input("count-plots-close-btn", "n_clicks")],
    [State("count-plots-modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open
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
        return [Lang['Local']], {'display': 'none'}, classname.replace('toggled-on', '')
    else:  # online usage
        return [Lang['Online']], {'display': 'block'}, classname+'toggled-on'

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
               Output('footage-selection-control', 'style'),
               Output("show-graphs-control", "style")],
              [Input("mode-switch", "value")],
              [State("dropdown-footage-selection", "value")])
def toggle_display(value,selected_video):
    if value:
        children = [
            html.Div(
                className='md-12 d-flex align-items-center flex-column',
                children=[
                    reusable.drag_drop_container('upload-image', 'drop-div', [Lang['Drag and Drop'], Lang['or'], Lang['Select Files']]
                                                 ),
                    html.Div(children=[html.Div(id='output-image-upload',
                                className='row d-flex align-items-center justify-content-center flex-row')]),
                    html.Div(children=[dcc.Loading( 
                        type="circle",
                        children=[html.Div(id='output-image-process',
                     className='d-flex flex-column', children=['']) ])
                     ])
                ])
        ]
    else:
        children = [
            html.Div(
                className='md-12 d-flex align-items-center flex-column',
                children=[
                    html.Div(
                        id='',
                        className='d-flex flex-column align-items-center justify-content-center',
                        children=[
                            html.Div(
                                className='mb-3',
                               
                                children=(
                                    html.H3(Lang['Preview'],className='text-center text-primary font-weight-bold'),
                                    player.DashPlayer(
                                        id='video-preview',
                                        url='http://localhost:8050/videos/{}'.format(selected_video),
                                        controls=True,
                                        playing=False,
                                        volume=1,
                                       
                                    )
                                 
                                
                                )

                            ),
                            html.Button(
                                id='process-video-button',
                                n_clicks=0,
                                className='btn btn-lg btn-outline-success',
                                children=[html.Span(Lang['Start processing video'], className='mr-2'), html.Span(
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
                    dcc.Loading(
                        type='circle',
                        children=[html.Div(id='output-video-process',
                                className='container')]
                    )
                ])
        ]
        # static.default_footage_section()
    return [Lang['Footage'] if not value else Lang['Still images']], children, {'display': 'block'} if not value else {'display': 'none'},{'display': 'block'} if not value else {'display': 'none'}

    # Footage Selection
@app.callback([Output("video-preview", "url")],
              [Input('dropdown-footage-selection', 'value')])
def select_footage(footage):
    print(footage)
     
    return ['http://localhost:8050/videos/{}'.format(os.path.basename(footage))]

    # Model selection
@app.callback([Output("dropdown-footage-selection", "options"),
                Output("dropdown-footage-selection", "value")],
              [Input("dropdown-model-selection", "value")],
              [State("mode-switch", "value")
              ])
def change_model(model_type,value):
    load_model(model_type)
    # if value:
    #     print('issue here')
    #     return [],[]
        
    list_vidoes = list(glob.glob(os.path.join(
            View.config.VIDEOS_DIR_PATH,'sparse videos' if model_type in ['mobileSSD', 'yolo'] else 'crowd videos', '*.mp4')))
    options=[{'label': os.path.basename(
                    el), 'value': el}
                for el in list_vidoes
            ]
    print(list_vidoes[0])
    return options,list_vidoes[0]

   # Show graphs toggle
@app.callback([Output("graph-switch-label", "children")],
              [Input("show-graphs-switch", "value")])
def toggle_show_graph(value):
    global SHOW_GRAPHS
    SHOW_GRAPHS=value
    return [Lang['Yes']] if value else [Lang['No']]


    # Upload image
@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'filename'), Input('upload-image', 'contents')])
def update_output(list_of_names, list_of_contents):
    global ONLINE_MODE,images_list
    images_list=[]
    if list_of_contents is not None:
        print('[INFO] uploading...')
        children = [html.H3(Lang['Preview'],className='ml-5 text-primary font-weight-bold flex-break'),html.Hr()]
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



    #Show graphs interval elapsed
@app.callback(Output("live-count-plot", "figure"),
              [Input("interval-show-graphs", "n_intervals")])
def update_count_plots(n):
    csv_file_path=os.path.join(ModelManager.outputs_path,'temp.csv')
    df=functions.read_existing_data(csv_file_path)
    df.dropna(subset = ["value"], inplace=True)
    xtext,ytext,title=Lang['Timestamp'],Lang['Count'],Lang['Live process plot']
    layout=dict(title={
        'text':title,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center'
        },
    xaxis_title=xtext,
    yaxis_title=ytext,
    hovermode="closest",
    transition={
        'easing':'quad-in-out'
    }
    )
    return go.Figure(data=go.Scatter(x=df.index.tolist(),y=df['value'].values.tolist()),layout=layout)
#     #Show graphs interval elapsed
# @app.callback([Output("week-count-plot", "figure"),
#                 Output("day-count-plot", "figure"),
#                 Output("hours-count-plot", "figure")],
#               [Input("interval-show-graphs", "n_intervals")])
# def update_count_plots(n):
#     csv_file_path=os.path.join(ModelManager.outputs_path,'results_history.csv')
#     # return functions.show_plots(functions.read_existing_data(csv_file_path))
#     # dfs=functions.show_plots(functions.read_existing_data(csv_file_path))
#     [df_2h,df_8h,df_1d,df_1w]=functions.show_plots(functions.read_existing_data(csv_file_path))
#     # print([df["value"].values.tolist() for df in dfs])
#     week_fig=go.Figure(data=go.Scatter(x=df_1w.index.tolist(),y=df_1w['value'].values.tolist(),text='gfjhfhgfjhgfjhfhgf',name='ooooooooooo'))
#     day_fig=go.Figure(data=go.Scatter(x=df_1d.index.tolist(),y=df_1d['value'].values.tolist()))
#     hours_fig=go.Figure(data=go.Scatter(x=df_2h.index.tolist(),y=df_2h['value'].values.tolist()))
    
#     return week_fig,day_fig,hours_fig

#     return  go.Figure({
#                 'data': [{'hoverinfo': 'x+text',
#                           'name': 'Counting history',
#                           'text': [f'{count}' for count in df['value'].values.tolist()],
#                           'type': 'bar',
#                           'x': df.index.tolist(),
#                         #   'marker': {'color': colors},
#                           'y': df["value"].values.tolist()} for df in dfs],
#                 'layout': {'showlegend': True,
#                            'autosize': True,
#                            'paper_bgcolor': 'rgb(249,249,249)',
#                            'plot_bgcolor': 'rgb(249,249,249)',
#                            'xaxis': {'automargin': True, 'tickangle': -45},
#                            'yaxis': {'automargin': True, 'title': {'text': 'Count'}}}
#                 } )

    
##############################################################################################
#           Buttons clicks event handlers
##############################################################################################

    # Start detection click


@app.callback([Output("output-image-process", "children"),
               Output("socket-errors-div", "children")],
              [Input("process-imgs-button", "n_clicks")],
              [State("dropdown-model-selection", "value"),
               State("output-image-process", "children")])
def process_frames(button_click, model_type, children):
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
            # if CLIENT_SOCKET is None:
            #     CLIENT_SOCKET = ClientSocket(reconnection=False)
            #     CLIENT_SOCKET.on('server-error', handler=server_error_response)
            #     CLIENT_SOCKET.on('send-image', handler=append_res_img)
            #     CLIENT_SOCKET.on('process-done', handler=response_received)
            # if not CLIENT_SOCKET.connected:
            #     CLIENT_SOCKET.connect(SERVER_URL)

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


@app.callback([Output("output-video-process", "children"),
                Output("display-plots-div", "children")],
              [Input("process-video-button", "n_clicks")],
              [State("dropdown-model-selection", "value"),
               State("dropdown-footage-selection", "value")])
def process_video(button_click, model_type, video_path):
    global server, server_thread, get_regions_params,SHOW_GRAPHS, SERVER_URL,CLIENT_SOCKET,ONLINE_MODE,path_video
    if button_click > 0:
        path_video=video_path
        print('Process server state : ',('not None' if server is not None else 'None'))
        if server_thread is not None:
            print('\t Thread alive ?: ',server_thread.isAlive())

        # def launch_subprocess(model_type, video_path, queue):
        #     global server
        #     from flask import request

        #     load_model(model_type)
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

        load_model(model_type)

        if ONLINE_MODE:
            if CLIENT_SOCKET is None:
                CLIENT_SOCKET = ClientSocket(reconnection=False)
            if not CLIENT_SOCKET.connected:
                CLIENT_SOCKET.connect(SERVER_URL)
            vs=cv2.VideoCapture(video_path)    
            frame = vs.read()
                #VideoStream returns a frame, VideoCapture returns a tuple
            frame = frame[1] if len(frame)>1 else frame
            (H, W) = frame.shape[:2]
            CLIENT_SOCKET.emit('init-process-video',{'model_type':model_type,'height':H,'width':W})
            socket_thread=StoppableThread(target=emit_by_frame,args=(video_path,model_type,))
            socket_thread.start()

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
                    global SHOW_GRAPHS,path_video
                
                    params={
                        'show':True,
                        'tang': float(get_regions_params()['tang']),
                        'b': int(float(get_regions_params()['b']))
                    } if get_regions_params() is not None else None
                    return Response(ModelManager.process_video(path_video,args={'regions_params':params,'log_counts':SHOW_GRAPHS,'log_count_fcn':functions.log_count}), mimetype='multipart/x-mixed-replace; boundary=frame')

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
                                className='fa fa-pencil-alt')], className='btn mr-5', title=Lang['Draw a split line']),
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
                            html.Img(src=('http://localhost:4000'if not ONLINE_MODE else SERVER_URL)+'/stream?t='+str(datetime.datetime.now()),
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
                            html.H3(Lang['Scene Edition'],className='text-primary text-center'),
                            html.P(Lang['Draw the regions you want to focus on'],className='text-secondary text-center')

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
        ],[] if not SHOW_GRAPHS else [
            dcc.Interval(
                id="interval-show-graphs",
                interval=1000,
                n_intervals=0
            ) if SHOW_GRAPHS else html.Div(),
            dcc.Graph(
                        id="live-count-plot",
                        style={'height': '55vh'}
                    ),
            # dcc.Graph(
            #             id="week-count-plot",
            #             style={'height': '55vh'}
            #         ),
            # dcc.Graph(
            #             id="day-count-plot",
            #             style={'height': '55vh'}
            #         ),
            # dcc.Graph(
            #             id="hours-count-plot",
            #             style={'height': '55vh'}
            #         )

        ]
    else: return [],[]    





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

def emit_by_frame(video_path,model_type):
    global CLIENT_SOCKET
    vs = cv2.VideoCapture(video_path)
    while True:
        frame = vs.read()
            #VideoStream returns a frame, VideoCapture returns a tuple
        frame = frame[1] if len(frame)>1 else frame
        if frame is None:
            break
        frame = imutils.resize(frame, width=500)
        frame=HTML_IMG_SRC_PREFIX+functions.numpy_to_b64(
            frame, model_type not in ['mobileSSD', 'yolo'])
        frame=frame.encode("utf-8").split(b";base64,")[1]
        CLIENT_SOCKET.emit('frame-upload',{'frame':frame})
        # CLIENT_SOCKET.emit('process-frame',{'frame':frame})
        key = cv2.waitKey(10) & 0xFF

        if key == ord("q"):
            break
    CLIENT_SOCKET.emit('process-video',{'frame':frame})
    print('[INFO] Reading video completed.')


def append_res_img(data):
    global res_img_list
    print('[INFO] A result received from server.')
    res_img_list.append(
        (data['id'], data['data'], data['count']))


def receive_result_image(data):
    return html.Div(className='row', children=[

        html.Div(children=[
                    html.Div(html.H4(Lang['Original'], className='muted'),
                             className="d-flex justify-content-center"),
                    html.Img(id='img-org-{}'.format(id),
                             src=HTML_IMG_SRC_PREFIX+(images_list[i].decode("utf-8")), style={
                        'width': '100%'})

                    ],
                 className='col-md justify-content-center animate__animated animate__fadeInRight'),
        html.Div(children=[
            html.Div(html.H4(Lang['Estimated count : ']+str(int((count+1)/100)),
                             className='muted'), className="d-flex justify-content-center"),
            html.Img(id='img-{}'.format(id),
                     src=encoded_img, style={
                        'width': '100%'})

        ],
            className='col-md justify-content-center animate__animated animate__fadeInRight')
    ])
