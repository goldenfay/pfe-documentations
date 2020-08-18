
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

class SensorsDashboardView(Component):
    layout = None
    df=None

    def __init__(self, app, config,url_params):
        self.config = config
        self.url_params=url_params
        super(SensorsDashboardView, self).__init__(app)

            

    def initialize(self, app):
        

        self.sensors_path=os.path.join(self.config.SENSORS_DEFAULT_BASE_PATH)

        if not os.path.exists(self.sensors_path):
            self.layout= error_layout('fa-question-circle','Databade path not found',
                             'It seems that we have a problem with the database directory')
            return
            # Grap dataframe from the specific .csv file
        # csv_file=os.path.join(self.sensors_path,'all.csv')
        # if not os.path.exists(csv_file):
        #     functions.construct_combined_results(self.sensors_path)
        
        All_dfs=functions.construct_combined_results(self.sensors_path)

        most_busy_days=[]
        most_busy_hours=[]
        crowded_sensors=[]
        data=[]
        min_date=None
        max_date=None
        max_crowd_number=0
        for sensor in list(All_dfs):
            df =All_dfs[sensor]
            df=df.resample('s').max()
            data.append(go.Scatter(
                x=df.index.tolist(),y=df['value'].values.tolist(),name=sensor
                ))

            if min_date is None or min_date> df.index.min():
                min_date=df.index.min()
            if max_date is None or max_date< df.index.max():
                max_date=df.index.max()
            if max(df['value'].values.tolist())>=max_crowd_number:
                max_crowd_number=max(df['value'].values.tolist())
                crowded_sensors.append(sensor)
                # Calculate most busy days (day name, date)
            sensor_most_busy_days=df.groupby(df.index.date).agg({'value': 'mean'})
            sensor_most_busy_days.index=pd.to_datetime(sensor_most_busy_days.index)
            sensor_most_busy_days=sensor_most_busy_days.sort_values('value',ascending=False)
            sensor_most_busy_days=sensor_most_busy_days[sensor_most_busy_days['value']>0]
            sensor_most_busy_days=list(zip(functions.index_to_list_date(sensor_most_busy_days.index.tolist()),sensor_most_busy_days.index.day_name().tolist()))
            
            most_busy_days+=sensor_most_busy_days
                # Calculate Peak hours 
            sensor_most_busy_hours=df.groupby(df.index.hour).agg({'value': 'mean'})
            sensor_most_busy_hours=sensor_most_busy_hours.sort_values('value',ascending=False)
            sensor_most_busy_hours=sensor_most_busy_hours[sensor_most_busy_hours['value']>0]
            sensor_most_busy_hours=['{}h'.format(el) for el in sensor_most_busy_hours.index.tolist()]
            most_busy_hours+=sensor_most_busy_hours

        most_busy_days=list(set(most_busy_days))
        most_busy_hours=list(set(most_busy_hours))
       
            # Define the full history graph of all sensors combined
        figure=go.Figure(data=data,layout=figure_layout,frames=[])

            # Define default filtering start and end date
        start_date=min_date#All_dfs.index.min()
        end_date=max_date#All_dfs.index.max()
        delta=end_date-start_date
        delta_hours=delta/np.timedelta64(1,'h')
        
        nbr_sensors=len([dirname for dirname in os.listdir(self.sensors_path) if os.path.isdir(os.path.join(self.sensors_path,dirname))])
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
                                        reusable.basic_outlined_stat_info_card('Total sensors number',nbr_sensors,'light',text_color='text-primary'),
                                        reusable.basic_outlined_stat_info_card('Capturing since',min_date,'light',text_color='text-success'),
                                        reusable.basic_outlined_stat_info_card('Total working hours',delta_hours,'light',text_color='text-warning'),
                                        


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
                                                            id='combined-filter-date-picker-range',
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
                                                                        dcc.Graph(id='combined-full-history-graph',className='container',figure=figure,responsive=True)

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
                            width=12
                        )
                        

                    ]
                ),
                dbc.Row(
                    className='mt-5',
                    children=[
                        dbc.Col(
                            className='d-flex flex-column justify-content-around',
                            children=[
                                dbc.CardDeck(
                                    [
                                            reusable.basic_outlined_stat_info_card('Most dense crowd in one moment',max_crowd_number,'light',text_color='text-info'),
                                            reusable.basic_outlined_stat_info_card('Most busy days',most_busy_days[:3],'light',text_color='text-info'),
                                            reusable.basic_outlined_stat_info_card('Peak hours',', '.join(most_busy_hours),'light',text_color='text-info'),
                                            reusable.basic_outlined_stat_info_card('Crowded zones',', '.join(crowded_sensors),'light',text_color='text-info')
                                        


                                    ]
                                )
                                

                            ],

                            width=12
                        )
                    ]

                ),
                # dbc.Row(
                #     className='mt-5',
                #     children=[
                #         dbc.Col(
                #             children=[
                #                 dbc.CardDeck(
                #                     [
                #                         reusable.basic_stat_plot_card(dcc.Graph(figure=week_fig,responsive=True)),
                #                         reusable.basic_stat_plot_card(dcc.Graph(figure=day_fig,responsive=True)),
                #                         reusable.basic_stat_plot_card(dcc.Graph(figure=hours_fig,responsive=True))


                #                     ]
                #                 )
                #             ],
                #             width=12
                #         )
      
                   

                #     ]
                # )
                
            ]
        )
        SensorsDashboardView.df=All_dfs



    


# @app.callback(
#     Output('full-history-graph', 'figure'),
#     [Input('filter-date-picker-range', 'start_date'),
#      Input('filter-date-picker-range', 'end_date')])
# def update_output(start_date, end_date):


#     start_date=SensorsDashboardView.df.index.min() if start_date is None else start_date
#     end_date=SensorsDashboardView.df.index.max() if end_date is None else end_date
     
#     if SensorsDashboardView.df is not None:
#         print(start_date,end_date,type(start_date))
#         df=SensorsDashboardView.df
#         df=df[np.logical_and(df.index<=end_date , df.index>=start_date)]
#         return go.Figure(data=go.Scatter(
#             x=df.index.tolist(),y=df['value'].values.tolist()
#         ),layout=figure_layout)
    