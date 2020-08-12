
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objs as go


import re,os,sys,glob,datetime,traceback,inspect
from flask import Flask, Response,request

from threading import Thread


currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
sys.path.append(currentdir)
    # User's modules
from modelmanager import ModelManager

from components.base import Component
import components.reusable as reusable
import components.static as static
import functions

from app import app
config = None


error_layout=lambda icon, title,subtitle: dbc.Container([dbc.Row([
                html.Div(children=[
                    html.H3([html.Span(className='fa '+icon+' mr-3'),title]),
                                html.P([subtitle],className='h4 text-secondary')
                ])               
                            ],className="h-100 d-flex justify-content-center align-items-center")
                            ],
            fluid=True,style={'height':'100vh'}) 

figure_layout=dict(title={
            'text': 'Full counting history',
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

class StatsView(Component):
    layout = None
    df=None

    def __init__(self, app, config,url_params):
        self.config = config
        self.url_params=url_params
        super(StatsView, self).__init__(app)

            

    def initialize(self, app):
        if not self.validate_params():
            self.layout= error_layout('fa-exclamation-triangle','Invalid Parameters',
                             'The request contains invalid query parameters.')
            return

        self.sensor_path=os.path.join(self.config.SENSORS_DEFAULT_BASE_PATH,self.url_params['sensor_name'][0])

        if not os.path.exists(self.sensor_path):
            self.layout= error_layout('fa-question-circle','Sensor not registred',
                             'The requested sensor Id does not exist')
            return
            # Grap dataframe from the specific .csv file
        csv_file=os.path.join(self.sensor_path,'output','temp.csv')
        df=functions.read_existing_data(csv_file)

        
        valuesAxes=df['value'].values
            # Calculate most busy days (day name, date)
        most_busy_days=df.groupby(df.index.date).agg({'value': 'mean'})
        most_busy_days.index=pd.to_datetime(most_busy_days.index)
        most_busy_days=list(zip(functions.index_to_list_date(most_busy_days.index.tolist()),most_busy_days.index.day_name().tolist()))
           
            # Calculate Peak hours 
        most_busy_hours=df.groupby(df.index.hour).agg({'value': 'max'})
        most_busy_hours=['{}h'.format(el) for el in most_busy_hours.index.tolist()]
        # most_busy_hours=list(zip(most_busy_hours.index.tolist(),most_busy_hours['value'].values.tolist()))

            # Get last week, last day, last 2 hours data and map them to correponding figures
        week_fig,day_fig,hours_fig=static.history_count_figures(csv_file)
            # Define Graph layout options
        
        splitpoint=len(df)//4
        parts=[splitpoint,2*splitpoint,3*splitpoint,len(df)]
        frames=[] if True else [
            go.Frame(data=[
                go.Scatter(x=df.index.tolist()[:idx],y=valuesAxes.tolist()[:idx])
            ])
            for idx in parts
        ]
            # Define the full history graph figure
        figure=go.Figure(data=go.Scatter(
            x=df.index.tolist(),y=valuesAxes.tolist()
        ),layout=figure_layout,frames=frames)

            # Define default filtering start and end date
        start_date=df.index.min()
        end_date=df.index.max()
        delta=end_date-start_date
        delta_hours=delta/np.timedelta64(1,'h')
        
        
        self.layout = dbc.Container(

            className='mt-5',
            children=[
                dbc.Row(
                    className='mt-5',
                    children=[
                        dbc.Col(
                            children=[
                                dbc.CardDeck(
                                    [
                                        reusable.basic_stat_info_card('Most busy day',most_busy_days,'primary'),
                                        reusable.basic_stat_info_card('Most dense crowd in one moment',max(valuesAxes.tolist()),'danger'),
                                        reusable.basic_stat_info_card('Peak hours',', '.join(most_busy_hours),'warning')


                                    ]
                                )
                            ],
                            width=12
                        )

                    ]
                ),
                dbc.Row(
                    className='mt-5',
                    children=[
                        dbc.Col(
                           
                            children=[
                                dbc.Card(
                                    children=[
                                        dbc.CardBody(
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.Span('Filter results betwwen : '),
                                                        dcc.DatePickerRange(
                                                            id='filter-date-picker-range',
                                                            min_date_allowed=datetime.datetime(start_date.year,start_date.month,start_date.day).date(),
                                                            max_date_allowed=datetime.datetime.now().date(),

                                                            # end_date=datetime.datetime(end_date.year,end_date.month,end_date.day).date()
                                                        )

                                                    ]
                                                ),
                                                dbc.Container(
                                                   
                                                    children=[dcc.Loading(
                                                            
                                                            type="circle",
                                                            children=[
                                                                dbc.Container(
                                                                        dcc.Graph(id='full-history-graph',className='container',figure=figure,responsive=True)

                                                                )
                                                            ]
                                                        )
                                                    ]
                                                )
                                                

                                            ]
                                        ),
                                        

                                    ]
                                )  

                            ],
                            width=7
                        ),
                        dbc.Col(
                            className='offset-md-1 d-flex flex-column justify-content-around',
                            children=[
                                dbc.Row(
                                    children=[
                                        dbc.Col(
                                            reusable.basic_stat_info_card('Total active hours',delta_hours,'info')
                                        )
                                            

                                    ]
                                ),
                                dbc.Row(
                                    children=[
                                        dbc.Col(
                                            reusable.basic_stat_info_card('Last capture','{}h'.format(end_date.hour),'success')
                                        )
                                            

                                    ]
                                )

                            ],

                            width=4
                        )

                    ]
                ),
                dbc.Row(
                    className='mt-5',
                    children=[
                        dbc.Col(
                            children=[
                                dbc.CardDeck(
                                    [
                                        reusable.basic_stat_plot_card(dcc.Graph(figure=week_fig,responsive=True)),
                                        reusable.basic_stat_plot_card(dcc.Graph(figure=day_fig,responsive=True)),
                                        reusable.basic_stat_plot_card(dcc.Graph(figure=hours_fig,responsive=True))


                                    ]
                                )
                            ],
                            width=12
                        )
      
                   

                    ]
                )
                
            ]
        )
        StatsView.df=df



    def validate_params(self):
        if not self.url_params.get('sensor_name',False):
            return False  

        return True     


@app.callback(
    Output('full-history-graph', 'figure'),
    [Input('filter-date-picker-range', 'start_date'),
     Input('filter-date-picker-range', 'end_date')])
def update_output(start_date, end_date):


    start_date=StatsView.df.index.min() if start_date is None else start_date
    end_date=StatsView.df.index.max() if end_date is None else end_date
     
    if StatsView.df is not None:
        print(start_date,end_date,type(start_date))
        df=StatsView.df
        df=df[np.logical_and(df.index<=end_date , df.index>=start_date)]
        return go.Figure(data=go.Scatter(
            x=df.index.tolist(),y=df['value'].values.tolist()
        ),layout=figure_layout)
    