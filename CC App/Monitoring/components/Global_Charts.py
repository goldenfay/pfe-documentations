import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import json
import components.reusable as reusable
from components.base import Component

class Global_Charts(Component):
    layout=None
    def __init__(self,app,charts_datafile):
        self.data_file=charts_datafile
        super(Global_Charts,self).__init__(app)
           
    def get_max_hour(self,data):
        max_crowd = max(data['crowd_data']['crowd_number'])
        hours = []
        i = 0
        for hour in data['crowd_data']['time']:
            if max_crowd == data['crowd_data']['crowd_number'][i]:
               hours.append(hour)
            i = i + 1   
        return hours
    
    def get_min_hour(self,data):
        min_crowd = min(data['crowd_data']['crowd_number'])
        hours = []
        i = 0
        for hour in data['crowd_data']['time']:
            if min_crowd == data['crowd_data']['crowd_number'][i]:
                hours.append(hour)
            i = i + 1    
        return hours
    
    def get_max_zone(data):
        max_zone = max(data['crowd_data']['zones'])
        '''max_zone is in the form of tuple (zone,crowd_nb,hour)'''
        return max_zone
    
    def get_min_zone(data):
        min_zone = min(data['crowd_data']['zones'])
        '''min_zone is in the form of tuple (zone,crowd_nb,hour)'''
        return min_zone
    
    def count_crowd_zone(self,data):
        temp_zone = data['crowd_data']['zones']
        temp_dict = {}
        for element in temp_zone:
            if element[0] in temp_dict:
                temp_dict[element[0]] = temp_dict[element[0]] + element[1]
            else:
                temp_dict[element[0]] = element[1]
        zone_list = []
        total_crowd = []
        for key in temp_dict.keys():
            zone_list.append(key)
            total_crowd.append(temp_dict[key])
        return zone_list,total_crowd    
    
    def initialize(self,app):
        '''Generated Pie Chart'''

        df = px.data.tips()
        fig = px.pie(df, values='tip', names='day')

        '''Opening JSON file''' 
        with open(self.data_file, 'r') as openfile: 
            data_points = json.load(openfile)

        '''Creating some Cards'''
        max_crowd = max(data_points['crowd_data']['crowd_number'])
        min_crowd = min(data_points['crowd_data']['crowd_number'])
        first_card = dbc.Card( reusable.params_card(self.get_max_hour(data_points),max_crowd,"Heure Max","Heure(s): ","Nombre de personnes: "))
    
        second_card = dbc.Card(reusable.params_card(self.get_min_hour(data_points),min_crowd,"Heure Min","Heure(s): ","Nombre de personnes: "), color="primary", inverse=True)
        
        third_card = dbc.Card(
            dbc.CardBody(
                [
                    html.H1("Card title", className="card-title"),
                    html.P(
                        "This card also has some text content and not much else, but "
                        "it is twice as wide as the first card.",className="h3"
                    ),
                    dbc.Button("Go somewhere", color="primary"),
                ]
            )
        )
        fourth_card = dbc.Card(
            dbc.CardBody(
                [
                    html.H1("Card title", className="card-title"),
                    html.P(
                        "This card also has some text content and not much else, but "
                        "it is twice as wide as the first card.",className="h3"
                    ),
                    dbc.Button("Go somewhere", color="primary"),
                ]
            )
        )
                
        '''Section dedicated for charts'''

        MyChart = html.Div(children=[
            html.H1(children='Graphe Statistique'),
            dcc.Graph(
                id='example-graph1',
                figure={
                    'data': [
                        {'x': data_points['crowd_data']['time'], 'y': data_points['crowd_data']['crowd_number'], 'type': 'line', 'name': 'Nombre de foule'}
                    ],
                    'layout': {
                        'title': 'Nombre de la foule dans les différentes heures'
                    }
                }
            )
        ])
        zones , total_crowd = self.count_crowd_zone(data_points)    
        BarChart = html.Div(children=[
            html.H1(children='Bar Chart'),

            html.Div(children='''
                Nombre total de la foule passé par la zone 'X'Dash Data Visualization.
            '''),

            dcc.Graph(
                id='example-graph',
                figure={
                    'data': [
                        {'x': zones, 'y': total_crowd, 'type': 'bar', 'name': 'SF'},
                    ],
                    'layout': {
                        'title': 'Nombre total de la foule passé par la zone \'X\''
                    }
                }
            )
        ])    
            
        BarChart2 = html.Div(children=[
            html.H1(children='Hello Dash'),

            html.Div(children='''
                Dash: A web application framework for Python.
            '''),

            dcc.Graph(
                id='example-graph2',
                figure= fig
            )
        ])     

            
            
        '''The LayOut Architecture'''    

        self.layout = html.Div(
            [
                
                dbc.Row([
                        dbc.Col(html.Div(first_card), lg=3),
                        dbc.Col(html.Div(second_card), lg=3),
                        dbc.Col(html.Div(first_card), lg=3),
                        dbc.Col(html.Div(second_card), lg=3),
                    ],
                    align="center"
                ),
                dbc.Row(
                    [
                        #dbc.Col(html.Div("One of three columns")),
                        dbc.Col(dbc.Col(MyChart)),
                        #dbc.Col(html.Div("One of three columns")),
                    ],
                    align="center",
                ),
                dbc.Row(
                    [
                        dbc.Col(dbc.Col(BarChart),width=6),
                        dbc.Col(dbc.Col(BarChart2),width=6),
                        #dbc.Col(html.Div("One of three columns")),
                    ],
                    align="end",
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Div("One of three columns"), align="start"),
                        dbc.Col(html.Div("One of three columns"), align="center"),
                        dbc.Col(html.Div("One of three columns"), align="end"),
                    ]
                ),
            ]
        )           
    
