from textwrap import dedent
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_player as player
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from matplotlib import cm
from io import BytesIO as _BytesIO
from PIL import Image
import re,time,base64,os,sys,glob,datetime,traceback,inspect
import multiprocessing

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(currentdir)
# User's modules
import components.static as static
import components.reusable as reusable

from modelmanager import ModelManager
from components.base import Component
from app import app
import functions


images_list = []
SERVER_URL = ''
ONLINE_MODE = False
HTML_IMG_SRC_PREFIX = 'data:image/png;base64, '
config = None
server=None



def parse_contents(contents, filename):
    global images_list
    images_list.append(contents.encode("utf-8").split(b";base64,")[1])
    return html.Div(children=[
        html.H5(filename, style={'textAlign': 'center'}),


        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents,id=re.sub('\.d+','',filename) ,style={
            'maxWidth': '80%', 'minWidth': '80px'})

    ],
        className='col-md justify-content-center "animate__animated animate__fadeInRight')

class View(Component):
    layout = None

    def __init__(self, app, config):
        self.config = config
        super(View, self).__init__(app)

    def initialize(self, app):
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
                                             'value': 'MCNN', 'disabled': True},
                                            {'label': 'Mobile SSD',
                                             'value': 'mobileSSD'},
                                            {'label': 'YOLO',
                                             'value': 'yolo'},
                                            {'label': 'Density map based models:',
                                             'value': 'MCNN', 'disabled': True},
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
                                        html.Div(id="div-detection-mode")
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
                    html.Div(id='output-image-upload',
                                className='row d-flex align-items-center justify-content-center flex-row')
                ])
        ]
    else:
        # children = [
        #     html.Div( style={"display":"none"}, #remove this later
        #         className='md-12 d-flex align-items-center flex-column',
        #         children=[
        #             reusable.drag_drop_container('upload-video','drop-div-video',['Click here','to','select a video']
        #             ),
        #             html.Div(id='output-video-upload',
        #                         className='row d-flex align-items-center justify-content-center flex-row')
        #         ])
        # ]
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
                                children=[html.Span('Start processing video',className='mr-2'), html.Span(
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
                                className='row d-flex align-items-center justify-content-center flex-row')
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
    print('Loading model : ', model_type, ' ...')
    if model_type in ['mobileSSD', 'yolo']:
        x = ModelManager.load_detection_model(model_type)
        print(type(x))
    else:
        try:
            ModelManager.load_external_model(model_type)
        except Exception as e:
            print('An error occured when loading model ',
                  model_type, end='\n\t')
            traceback.print_exc()
            pass
    print('Done.')
    return {"display": "none"}

    # Upload image
@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'filename'), Input('upload-image', 'contents')])
def update_output(list_of_names, list_of_contents):
    global ONLINE_MODE

    if list_of_contents is not None:
        print('uploading...')
        children = []
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
            html.Div(id='output-image-process',
                     className='d-flex flex-row align-items-center', children=[''])
        ]

        print('Done')
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


@app.callback(Output("output-image-process", "children"),
              [Input("process-imgs-button", "n_clicks")],
              [State("dropdown-model-selection", "value"),
               State("output-image-process", "children")])
def start_detection(button_click, model_type, children):
    global images_list
    return []
    if button_click > 0:
        frames = [functions.b64_to_numpy(el) for el in images_list]

        # results=ModelManager.process_frame(frames)
        # res_img_list=[]
        # for img,count in enumerate(results):
        #     print('Count : ', count)
        #     encoded_img = numpy_to_b64(img,model_type not in ['mobileSSD','yolo'])
        #     res_img_list.append(encoded_img)

        res_img_list = []
        for id, frame in enumerate(frames):
            print('Processing image :\n\t frame of shape : ', frame.shape)
            try:

                start = time.time()
                dmap, count = ModelManager.process_frame(frame)
                inference_time = time.time()-start

                print('\t Inference time : ', inference_time, 'count : ', count)
                encoded_img = functions.numpy_to_b64(
                    dmap, model_type not in ['mobileSSD', 'yolo'])
                res_img_list.append(
                    (id, HTML_IMG_SRC_PREFIX+encoded_img, count))
            except Exception as e:
                print("An error occured while detecting ", end='\n\t')
                traceback.print_exc()

        return [
            html.Div(className='row', children=[

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
            for (i, (id, encoded_img, count)) in enumerate(res_img_list)]

    # Process video button click
@app.callback(Output("output-video-process", "children"),
              [Input("process-video-button", "n_clicks")],
              [State("dropdown-model-selection", "value"),
               State("dropdown-footage-selection", "value")])
def process_video(button_click, model_type, video_path):
    global server
    if button_click > 0:
      
        ModelManager.process_video(video_path)

            # run=lambda :server.run(port=4000)
            # multiprocessing.Process(target=run).start()

        return [
            html.Img(src='http://localhost:4000/video_feed')

        ]


# Learn more popup
@app.callback(Output("markdown", "style"),
              [Input("learn-more-button", "n_clicks"), Input("markdown_close", "n_clicks")])
def update_click_output(button_click, close_click):
    if button_click > close_click:
        return {"display": "block"}
    else:
        return {"display": "none"}

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

# @server.route('/video_feed')
# def video_feed():
#     return Response(gen(VideoCamera()),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')