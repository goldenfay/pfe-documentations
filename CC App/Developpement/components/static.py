"""
    This module is dedicated to build static components for different GUIs
    
"""

import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_player as player
from textwrap import dedent
import plotly.graph_objs as go
    # User's modules
import functions


Lang=None


def default_header():
    return html.Div(
        id='header-section',
        className="md-12",
        children=[
            html.H4(
                'Crowd counting Monitor'
            ),
            html.P(
                Lang['This is a developement application to manage crowd counting processes']
            ),
            html.Button(Lang['Learn More'], id="learn-more-button", className='border-0',
                        n_clicks=0)
        ]
    )

def markdown_popup():
    return html.Div(
        id='markdown',
        className="model",
        style={'display': 'none'},
        children=(
            html.Div(
                className="markdown-container",
                children=[
                    html.Div(
                        className='close-container',
                        children=html.Button(
                            Lang['Close'],
                            id="markdown_close",
                            n_clicks=0,
                            className="closeButton",
                            style={'border': 'none', 'height': '100%'}
                        )
                    ),
                    html.Div(
                        className='markdown-text',
                        children=[dcc.Markdown(
                            children=dedent(
                                '''

                                ##### More about this application
                                
                                This is a developement app for models and functionnalities testing.
                                The main purpose of this application is to enhance developement cycle phases and to get an idea about how the deployement application may look.
                                This application allows us to gain time and work incrementally by exploiting each model and evaluate its performances. Il offers also the possibility
                                to emulate a real sensor behaviour while capturing scences. 
                                 
                                '''
                            ))
                        ]
                    )
                ]
            )
        )
    )

model_selection_options= lambda :[
                    {'label': Lang['Detection models:'],
                        'value': 'DM', 'disabled': True},
                    {'label': 'Mobile SSD',
                        'value': 'mobileSSD'},
                    {'label': 'YOLO',
                        'value': 'yolo'},
                    {'label': Lang['Density map based models:'],
                        'value': 'CNCC', 'disabled': True},
                    {'label': 'MCNN',
                        'value': 'MCNN'},
                    {'label': 'CSRNet',
                        'value': 'CSRNet'},
                    {'label': 'SANet',
                        'value': 'SANet'}
                ]
def default_footage_section():
    return [html.Div(
        className='video-outer-container',
        children=html.Div(
            style={'width': '100%', 'paddingBottom': '56.25%',
                   'position': 'relative'},
            children=[
            #     player.DashPlayer(
            #     id='video-display',
            #     style={'position': 'absolute', 'width': '100%',
            #            'height': '100%', 'top': '0', 'left': '0', 'bottom': '0', 'right': '0'},
            #     url='https://www.youtube.com/watch?v=gPtn6hD7o8g',
            #     controls=True,
            #     playing=False,
            #     volume=1,
            #     width='100%',
            #     height='100%'
            # )
            ]
        )
    ),
        html.Div(
        className='video-outer-container',
        children=html.Div(
            style={'width': '100%', 'paddingBottom': '56.25%',
                   'position': 'relative'},
            children=[
            #     player.DashPlayer(
            #       id='video-display2',
            #       style={'position': 'absolute', 'width': '100%',
            #              'height': '100%', 'top': '0', 'left': '0', 'bottom': '0', 'right': '0'},
            #       url='https://www.youtube.com/watch?v=gPtn6hD7o8g',
            #       controls=True,
            #       playing=False,
            #       volume=1,
            #       width='100%',
            #       height='100%'
            # )
            ]
        )
    )
    ]
def history_count_figures(csv_file_path):
    xtext,ytext='Timestamp','Count'
    layout=dict(title={
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
    [df_2h,df_8h,df_1d,df_1w]=functions.show_plots(functions.read_existing_data(csv_file_path))
    week_fig=go.Figure(data=go.Scatter(x=df_1w.index.tolist(), y=df_1w['value'].values.tolist()),layout=layout)
    day_fig=go.Figure(data=go.Scatter(x=df_1d.index.tolist(),y=df_1d['value'].values.tolist()),layout=layout)
    hours_fig=go.Figure(data=go.Scatter(x=df_2h.index.tolist(),y=df_2h['value'].values.tolist()),layout=layout)
    
    week_fig.update_layout({
        'title': {'text':Lang['Last week''s counting analytics']}
    })
    day_fig.update_layout({
        'title': {'text':Lang['Last day counting analytics']}
    })
    hours_fig.update_layout({
        'title': {'text':Lang['Last 2 hours counting analytics']}
    })
    return week_fig,day_fig,hours_fig

def default_count_plots_modal(csv_file_path):
    week_fig,day_fig,hours_fig=history_count_figures(csv_file_path)
    
    return html.Div(
    [
        dbc.Button(Lang['View history'], id="view-count-plots-btn"),
        dbc.Modal(
            [
                dbc.ModalHeader(html.Div("History Analytics",className="text-center")),
                dbc.ModalBody(children=[
                    html.Div(className='row',
                    children=[
                        html.Div(className="col-md",children=[dcc.Graph(figure=week_fig)]),
                        html.Div(className="col-md",children=[dcc.Graph(figure=day_fig)]),
                        html.Div(className="col-md",children=[dcc.Graph(figure=hours_fig)])

                    ])
                    
                ]),
                dbc.ModalFooter(
                    dbc.Button(
                        Lang['Close'], id="count-plots-close-btn", className="ml-auto"
                    )
                ),
            ],
            id="count-plots-modal",
            size="xl",
            centered=True,
        ),
    ]
)
