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
import re,time,base64,sys,datetime,traceback
# User's modules
import components.static as static
import components.reusable as reusable
# from ... import modelmanager
from modelmanager import ModelManager
from components.base import Component
from app import app
global images_list
images_list = []
HTML_IMG_SRC_PARAMETERS = 'data:image/png;base64, '
config = None


def b64_to_pil(string):
    decoded = base64.b64decode(string)
    buffer = _BytesIO(decoded)
    im = Image.open(buffer)

    return im


def b64_to_numpy(string, to_scalar=True):
    im = b64_to_pil(string)
    np_array = np.asarray(im)[:, :, :3]

    if to_scalar:
        np_array = np_array / 255.

    return np_array


def numpy_to_pil(array, jetMap=True):
    if jetMap:
        print('\t Converting to Jet color map')
        array = cm.jet(array)
    return Image.fromarray(np.uint8(array*255))


def numpy_to_b64(array, jetMap=True):
    im_pil = numpy_to_pil(array, jetMap)
    buff = _BytesIO()
    im_pil.save(buff, format="png")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def pil_to_b64(im, enc_format='png', verbose=False, **kwargs):

    buff = _BytesIO()
    im.save(buff, format=enc_format, **kwargs)
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")
    return encoded


def load_data(path):
    """Load data about a specific footage (given by the path). It returns a dictionary of useful variables such as
    the dataframe containing all the detection and bounds localization, the number of classes inside that footage,
    the matrix of all the classes in string, the given class with padding, and the root of the number of classes,
    rounded."""

    # Load the dataframe containing all the processed object detections inside the video
    video_info_df = pd.read_csv(path)

    # The list of classes, and the number of classes
    classes_list = video_info_df["class_str"].value_counts().index.tolist()
    n_classes = len(classes_list)

    # Gets the smallest value needed to add to the end of the classes list to get a square matrix
    root_round = np.ceil(np.sqrt(len(classes_list)))
    total_size = root_round ** 2
    padding_value = int(total_size - n_classes)
    classes_padded = np.pad(classes_list, (0, padding_value), mode='constant')

    # The padded matrix containing all the classes inside a matrix
    classes_matrix = np.reshape(
        classes_padded, (int(root_round), int(root_round)))

    # Flip it for better looks
    classes_matrix = np.flip(classes_matrix, axis=0)

    data_dict = {
        "video_info_df": video_info_df,
        "n_classes": n_classes,
        "classes_matrix": classes_matrix,
        "classes_padded": classes_padded,
        "root_round": root_round
    }

    # if DEBUG:
    #     print(f'{path} loaded.')

    return data_dict


def parse_contents(contents, filename):
    images_list.append(contents.encode("utf-8").split(b";base64,")[1])
    return html.Div(children=[
        html.H5(filename, style={'text-align': 'center'}),


        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents, style={
            'maxWidth': '80%', 'minWidth': '80px'})

    ],
        className='col-md justify-content-center "animate__animated animate__fadeInRight')


class View(Component):
    layout = None

    def __init__(self, app, config):
        self.config = config
        super(View, self).__init__(app)

    def initialize(self, app):

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

                                'overflow-y': 'scroll',

                                'backgroundColor': '#F9F9F9'
                            },
                            children=[
                                html.Div(
                                    className='control-section d-flex flex-column',
                                    children=[
                                        html.Div(
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
                                                    'text-align': 'center'}
                                                )
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
                                        reusable.dropdown_control("Footage Selection:", [
                                            {'label': 'Drone recording of canal festival',
                                             'value': 'DroneCanalFestival'},
                                            {'label': 'Drone recording of car festival',
                                             'value': 'car_show_drone'},
                                            {'label': 'Drone recording of car festival #2',
                                             'value': 'DroneCarFestival2'},
                                            {'label': 'Drone recording of a farm',
                                             'value': 'FarmDrone'},
                                            {'label': 'Lion fighting Zebras',
                                             'value': 'zebra'},
                                            {'label': 'Man caught by a CCTV',
                                             'value': 'ManCCTV'},
                                            {'label': 'Man driving expensive car',
                                             'value': 'car_footage'},
                                            {'label': 'Restaurant Robbery',
                                             'value': 'RestaurantHoldup'}
                                        ],
                                            'car_show_drone',
                                            id="dropdown-footage-selection"


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

# Switching layouts

@app.callback([Output("switch-label", "children"),
               Output('footage-container', 'children')],
              [Input("mode-switch", "value")])
def toggle_display(value):
    if value:
        children = [
            html.Div(
                className='md-12 d-flex align-items-center flex-column',
                children=[
                    html.Div(
                        children=[dcc.Upload(
                            id='upload-image',
                            className='d-flex align-items-center justify-content-center',
                            children=html.Div(
                                id='drop-div',
                                className='align-self-center',
                                children=[
                                    html.Div(['Drag and Drop']),
                                    html.Div(['or']),
                                    html.Div([html.A('Select Files')])
                                ]),
                            style={
                                'width': '500px',
                                'height': '300px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',

                            },

                            # Allow multiple files to be uploaded
                            multiple=True
                        )]
                    ),
                    html.Div(id='output-image-upload',
                                className='row d-flex align-items-center justify-content-center flex-row')
                ])
        ]
    else:
        children = [
            html.Div(
                className='md-12 d-flex align-items-center flex-column',
                children=[
                    html.Div(
                        children=[dcc.Upload(
                            id='upload-video',
                            className='d-flex align-items-center justify-content-center',
                            children=html.Div(
                                id='drop-div-video',
                                className='align-self-center',
                                children=[
                                    html.Div(['Click here']),
                                    html.Div(['to']),
                                    html.Div([html.A('Select Video')])
                                ]),
                            style={
                                'width': '500px',
                                'height': '300px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',

                            },

                            # Allow multiple files to be uploaded
                            multiple=True
                        )]
                    ),
                    html.Div(id='output-video-upload',
                                className='row d-flex align-items-center justify-content-center flex-row')
                ])
        ]
        #static.default_footage_section()
    return ['Footage' if not value else 'Still images'], children

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
    if list_of_contents is not None:
        print('uploading...')
        children = []
        for i in range(len(list_of_contents)):
            children.append(parse_contents(
                list_of_contents[i], list_of_names[i]))

        children += [html.Div(children=[html.Button(
            id='process-imgs-button',
            n_clicks=0,
            className='btn btn-md btn-outline-success',
            children=[html.Span(className='fa fa-play')])],
            style={'font-weight': 'bold',
                   'font-size': '13px',
                   'min-width':'20px'}
        ),
            html.Div(id='output-image-process',
                     className='d-flex flex-row align-items-center', children=[''])
        ]

        print('Done')
        return children

        #Upload video

@app.callback(
    Output('output-video-upload', 'children'),
    [Input("upload-video", "filename"), Input("upload-video", "contents")],
)
def update_output(uploaded_filenames, uploaded_file_contents):
        #if uploaded_filenames is not None and uploaded_file_contents is not None:
            var=""
            if uploaded_filenames is not None :
                var = uploaded_filenames[0]   
                return html.Div(html.Video(id="myvideo",controls = True,src='http://127.0.0.1:8080/'+var, style={'width': '500px','height': '300px'}))     

    # Start detection click


@app.callback(Output("output-image-process", "children"),
              [Input("process-imgs-button", "n_clicks")],
              [State("dropdown-model-selection", "value"),
               State("output-image-process", "children")])
def start_detection(button_click, model_type, children):
    if button_click > 0:
        frames = [b64_to_numpy(el) for el in images_list]

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
                encoded_img = numpy_to_b64(
                    dmap, model_type not in ['mobileSSD', 'yolo'])
                res_img_list.append((id, HTML_IMG_SRC_PARAMETERS+encoded_img,count))
            except Exception as e:
                print("An error occured while detecting ", end='\n\t')
                traceback.print_exc()

        return [
            html.Div(className='row',children=[

                html.Div(children=[
                    html.Div(html.H4('Original',className='muted'),className="d-flex justify-content-center"),
                    html.Img(id='img-org-{}'.format(id),
                        src=HTML_IMG_SRC_PARAMETERS+(images_list[i].decode("utf-8")), style={
                        'width':'100%'})

                ],
                    className='col-md justify-content-center animate__animated animate__fadeInRight'),
                html.Div(children=[
                    html.Div(html.H4('Estimated count : '+str(int((count+1)/100 ) ),className='muted'),className="d-flex justify-content-center"),
                    html.Img(id='img-{}'.format(id),
                        src=encoded_img, style={
                        'width':'100%'})

                ],
                    className='col-md justify-content-center animate__animated animate__fadeInRight')
            ])
            for (i,(id, encoded_img,count)) in enumerate(res_img_list)]


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
