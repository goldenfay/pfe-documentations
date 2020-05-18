import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import json
from components.base import Component

class Global_Charts(Component):
    layout=None
    def __init__(self,app,charts_datafile):
        self.data_file=charts_datafile
        super(Global_Charts,self).__init__(app)

    def initialize(self,app):
        '''Generated Pie Chart'''

        df = px.data.tips()
        fig = px.pie(df, values='tip', names='day')

        '''Opening JSON file''' 
        with open(self.data_file, 'r') as openfile: 
            data_points = json.load(openfile)

        '''Creating some Cards'''

        first_card = dbc.Card(
            dbc.CardBody(
                [
                    html.H1("Paramètre", className="card-title"),
                    html.P("Affichage du contenu de paramètre",className="h3"),
                ]
            )
        )
        second_card = dbc.Card(
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

            html.Div(children='''
                Simple exemple montre le nombre de la foule par heure.
            '''),

            dcc.Graph(
                id='example-graph1',
                figure={
                    'data': [
                        {'x': data_points['X'], 'y': data_points['Y'], 'type': 'line', 'name': 'Nombre de foule'}
                    ],
                    'layout': {
                        'title': 'Nombre de la foule dans les différentes heures'
                    }
                }
            )
        ])
            
        BarChart = html.Div(children=[
            html.H1(children='Hello Dash'),

            html.Div(children='''
                Dash: A web application framework for Python.
            '''),

            dcc.Graph(
                id='example-graph',
                figure={
                    'data': [
                        {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                        {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
                    ],
                    'layout': {
                        'title': 'Dash Data Visualization'
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
                dbc.Row(
                    [
                        dbc.Col(html.Div(first_card), width=3),
                        dbc.Col(html.Div(first_card), width=3),
                        dbc.Col(html.Div(first_card), width=3),
                        dbc.Col(html.Div(first_card), width=3),
                    ],
                    align="start",
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


