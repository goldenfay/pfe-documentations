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
from flask import Flask, Response, request
import requests
import threading


currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
sys.path.append(currentdir)
    # User's modules
from modelmanager import ModelManager,QUEUE

from components.base import Component
import components.reusable as reusable
import components.static as static
import functions
from threads import *

from app import app,get_regions_params



is_detectionModel= lambda model: model.lower() in ['mobilessd','yolo']
def kill_server_thread(serv_th):
    try:
        serv_th.raise_exception()
        x=requests.post('http://localhost:4000/shutdown')
        print(x)
        

    except:
        print('[Server] Server killed.')
        pass




def load_model(model_type):
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

Lang=None
model_type=None
sensor_path=None
server = None
server_thread = None
SHOW_LIVE_GRAPH=False
LIVE_DF=None
best_performence_models = {
    'MCNN': 'internal',
    'CSRNet': 'external',
    'SANet': 'external',
    'CCNN': 'internal'
}

error_layout=lambda icon, title,subtitle: dbc.Container([dbc.Row([
                html.Div(children=[
                    html.H3([html.Span(className='fa '+icon+' mr-3'),title]),
                                html.P([subtitle],className='h4 text-secondary')
                ])               
                            ],className="h-100 d-flex justify-content-center align-items-center")
                            ],
            fluid=True,style={'height':'100vh'}) 

figure_layout=lambda :dict(title={
            'text': Lang['Full counting history'],
            'y':0.9,
            'x':0.5,
            'xanchor': 'center'
            },
        xaxis_title='Timestamp',
        yaxis_title='Count',
        hovermode="closest",
        transition={
            'easing':'quad-in-out'
        }
        )


class SensorProcessView(Component):
    layout = None
    server = None

    def __init__(self, app, config,url_params):
        self.config = config
        self.url_params=url_params
        super(SensorProcessView, self).__init__(app)

    def reset_variables(self):
        global server, server_thread,LIVE_DF
        LIVE_DF = pd.DataFrame()
        LIVE_DF['timestamp'] = pd.Series(dtype='datetime64[ns]')
        LIVE_DF['value'] = pd.Series(dtype=np.int32)

        if server_thread is not None and server_thread.isAlive():
            kill_server_thread(server_thread)
        if server is not None:
            server=None    

    def validate_params(self):
            
        if not self.url_params.get('sensor_name',False):
            return False  
        if not self.url_params.get('model_name',False):
            return False  

        return True
    def initialize(self, app):
        global sensor_path,model_type,Lang
        Lang=self.config.LANGUAGE_DICT
        static.Lang=Lang
        self.reset_variables()
        if not self.validate_params():
            self.layout= error_layout('fa-exclamation-triangle',Lang['Invalid Parameters'],
                             Lang['The request contains invalid query parameters.'])
            return

        sensor_path=os.path.join(self.config.SENSORS_DEFAULT_BASE_PATH,self.url_params['sensor_name'][0])

        if not os.path.exists(sensor_path):
            self.layout= error_layout('fa-question-circle',Lang['Sensor not registred'],
                             Lang['The requested sensor Id does not exist.'])
            return
        model_type=self.url_params['model_name'][0]
        
        list_vidoes = list(glob.glob(os.path.join(
            self.config.VIDEOS_DIR_PATH,'sparse videos' if is_detectionModel(model_type) else 'crowd videos', '*.mp4')))

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
                                html.Div(
                                    id='display-container',
                                    children=[
                                        html.Div(
                                            className='md-12 d-flex align-items-center flex-column',
                                            children=[
                                                
                                                html.Div(
                                                    className='d-flex flex-column align-items-center justify-content-center',
                                                    children=[
                                                        html.Div(
                                                            
                                                            children=(
                                                                html.H3(Lang['Preview'],className='text-center text-primary font-weight-bold'),
                                                                player.DashPlayer(
                                                                    id='sensor-video-preview',
                                                                    # style={'position': 'absolute', 'width': '100%',
                                                                    #     'height': '100%', 'top': '0', 'left': '0', 'bottom': '0', 'right': '0'},
                                                                    url='http://localhost:8050/videos/{}'.format(os.path.basename(list_vidoes[0])),
                                                                    controls=True,
                                                                    playing=False,
                                                                    volume=1,
                                                                    width='100%',
                                                                    height='100%'
                                                                )
                                                                # html.Video(
                                                                #     id='sensor-video-preview',
                                                                #     src='http://localhost:8050/videos/{}'.format(os.path.basename(list_vidoes[0])),
                                                                #     controls=True,autoPlay=False
                                                                # )
                                                            
                                                            )

                                                        ),
                                                        html.Button(
                                                            id='start-process-button',
                                                            n_clicks=0,
                                                            className='btn btn-lg btn-outline-success',
                                                            children=[html.Span(Lang['Start '], className='mr-2'), html.Span(
                                                                className='fa fa-play')],
                                                            style={'fontWeight': 'bold',
                                                                   'fontSize': '26px',
                                                                   'minWidth': '100px',
                                                                   'minHeight': '60px'}

                                                        )
                                                    ]
                                                ),
                                                html.Div(id='sensor-output-video-process',
                                                         className='container')
                                            ])
                                    ],
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
                                            className='row mt-5',
                                            children=[
                                                html.Div(
                                                    className='col-md-12 d-flex justify-content-center',
                                                    children=[
                                                        html.H4(
                                                            Lang['Process settings'], className='muted')
                                                    ]
                                                )
                                            ]
                                        ),
                                        reusable.toggleswitch_control(Lang['Video source'],'video-source-switch','video-source-switch-label',True,Lang['Sample video'],'#8e24aa'),
                                        # reusable.toggleswitch_control('Show density map','show-dmap-switch','show-dmap-switch-label',True,Lang['Yes'],'#2196f3'),
                                        
                                        
                                        html.Div(children=reusable.dropdown_control(Lang['Footage Selection:'], [
                                            {'label': os.path.basename(
                                                el), 'value': el}
                                            for el in list_vidoes
                                        ],
                                            list_vidoes[0],
                                            id="sensor-dropdown-footage-selection"


                                        ),
                                            id='sensor-footage-selection-control'
                                        ),
                                        reusable.toggleswitch_control(Lang['Show live graph'],'show-live-graph-switch','show-live-graph-switch-label',True,Lang['Yes'],'#00c853')
                                        ,
                                        html.Div(id="sensor-display-plots-div",children=[]),
                                        
                                    ]
                                )]),

                    ]
                ),
                html.Script(src='http://localhost:8050/assets/js***sensor-process.js')
            ]
        )




##############################################################################################
#           Switch event handlers
##############################################################################################


@app.callback([Output("sensor-footage-selection-control", "style"),
                Output("video-source-switch-label", "children")],
                [Input("video-source-switch","value")]
            )
def toggle_video_source(value):
    if value:
        return {'display':'block'},[Lang['Sample video']]
    else:
        return {'display':'none'},[Lang['Sensor webcam']]     


@app.callback([Output("show-live-graph-switch-label", "children")],
                [Input("show-live-graph-switch","value")]
            )
def toggle_show_live_plot(value):
    global SHOW_LIVE_GRAPH
    SHOW_LIVE_GRAPH=value
    
    return [Lang['Yes']] if value else [Lang['No']]
    


##############################################################################################
#           Dropdowns event handlers
##############################################################################################

@app.callback([Output("sensor-video-preview", "url")],
                [Input("sensor-dropdown-footage-selection","value")]
            )
def change_video_footage(value):
    return ['http://localhost:8050/videos/{}'.format(os.path.basename(value))]


##############################################################################################
#           Buttons clicks event handlers
##############################################################################################

    # Process video button click


@app.callback([Output("sensor-output-video-process", "children"),
               Output("sensor-display-plots-div", "children")],
              [Input("start-process-button", "n_clicks")],
              [State("sensor-dropdown-footage-selection", "value")])
def process_video(button_click, video_path):
    global model_type,server, server_thread, get_regions_params
    if button_click > 0:
        
        load_model(model_type)

        if server is None:
            print('[SERVER] Creating a server instance ...')
            from flask import request
            server = Flask('SensorProcessServer')

            @server.route('/test', methods=['GET'])
            def test():
                return 'jfhskdjfhskjdhfkjshdkjfhskdjhf'

            @server.route('/shutdown', methods=['POST'])
            def shut_it():
                raise RuntimeError('Flask server killed via request')

            
            @server.route('/stream')
            def video_feed():
                global SHOW_LIVE_GRAPH,QUEUE

                params = {
                    'show': True,
                    'tang': float(get_regions_params()['tang']),
                    'b': int(float(get_regions_params()['b'])),
                    
                } if get_regions_params() is not None else None
                return Response(ModelManager.process_video(
                    video_path, 
                    args={'output': os.path.join(sensor_path,'output'),
                        'regions_params': params, 
                        'live_data': SHOW_LIVE_GRAPH, 
                        'log_counts': True, 
                        'log_count_fcn': functions.log_count},queue=QUEUE), mimetype='multipart/x-mixed-replace; boundary=frame')

            try:
                server_thread = ServerThread(server)
            except OSError as e:
                pass
                # requests.post('http://localhost:4000/shutdown')


        server_thread.start()

        return [


            html.Div(
                className='row shadow-sm',
                children=[
                    html.Div(
                        className='col-md-12 d-flex justify-content-center align-items-center',
                        id='sensor-edit-canvas-panel',
                        children=[
                            html.Button(id='sensor-line-canvas-button', children=[html.Span(
                                className='fa fa-pencil-alt')], className='btn mr-5', title='Draw a separation line'),
                            html.Button(children=[html.Span(
                                className='fa fa-times')], className='btn ml-5', id='sensor-clear-canvas-button', title='Cancel')

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
                            html.Img(src='http://localhost:4000/stream?t='+str(datetime.datetime.now()),
                                     id='sensor-process-video-output-flow',
                                     className='img-fluid'),
                        ]
                    )


                ]
            ),

            html.Div(
                id='sensor-edit-canvas-area',
                className='row d-none mt-5',
                children=[
                    html.Hr(
                        className='divider silent mt-5 mb-5'
                    ),
                    html.Div(
                        className='col-md-12 mt-5 d-flex flex-column justify-content-center',
                        children=[
                            html.H3(Lang['Scene Edition'],
                                    className='text-primary text-center'),
                            html.P(Lang['Draw the regions you want to focus on'],
                                   className='text-secondary text-center')

                        ]
                    ),
                    html.Div(
                        className='col-md-12 mt-5 d-flex justify-content-center align-items-center',
                        children=[html.Canvas(id='sensor-scene-canvas')]
                    ),
                    html.Div(
                        className='col-md-12 mt-5 shadow-sm d-flex justify-content-center align-items-center',
                        children=[
                            html.Button(id='sensor-confirm-draw-btn',n_clicks=0, className='btn-success', children=[
                                        html.Span(className='fa fa-check')], title='Confirm zones plit')
                        ]
                    )
                ]
            ),
            html.P(id='sensor-hidden-splitLine-input',
                      className='d-none', title='')
        ], [] if not SHOW_LIVE_GRAPH else [
            dcc.Interval(
                id="sensor-interval-show-graphs",
                interval=1000,
                n_intervals=0
            ) if SHOW_LIVE_GRAPH else html.Div(),
            dcc.Graph(
                id="sensor-live-count-plot",
                style={'height': '55vh'}
            )

        ]
    else:
        return [], []


@app.callback(Output("sensor-live-count-plot", "figure"),
              [Input("sensor-interval-show-graphs", "n_intervals")])
def update_count_plots(n):
    global sensor_path,LIVE_DF
    df=None
    try:
        # csv_file_path=os.path.join(sensor_path,'output','temp.csv')
        # df=functions.read_existing_data(csv_file_path)
        if not QUEUE.empty():
            print('Get from the que'+'*'*50)
            LIVE_DF=LIVE_DF.append(QUEUE.get_nowait(),ignore_index=True)
            # LIVE_DF=df.copy()
            
        # else:
        #     df=LIVE_DF 
        # dtime=pd.to_datetime(df['timestamp'])
        fcn=lambda x: x.date.__str__()+' '+str(x.hour)+':'+str(x.minute)+':'+str(x.second)
        # df=df.groupby(dtime.apply(fcn)).mean()#agg({'value': 'mean'})       
        df=LIVE_DF.set_index('timestamp')
        df=df.resample('s').max()#agg({'value': 'mean'}) 
        df.dropna(subset = ["value"], inplace=True)
        # print(df,df.index)
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
    except:
        traceback.print_exc()


@app.callback(
    Output("sensor-hidden-splitLine-input", "className"),
    [Input("sensor-confirm-draw-btn", "n_clicks")]
)
def setup_splitlines(n_clicks):
    global server, server_thread

    if n_clicks > 0 :
        import requests
        kill_server_thread(server_thread)
       
        server_thread.shutdown() 
        #server = None
    return 'd-none'
