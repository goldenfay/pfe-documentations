import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import json, csv
import components.reusable as reusable
import numpy as np
import pandas as pd
from components.base import Component

class Global_Charts(Component):
    layout=None
    def __init__(self,app,charts_datafile):
        self.data_file=charts_datafile
        super(Global_Charts,self).__init__(app)
           
        '''  THE JSON MODE IS HERE  '''

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
    
    def get_max_zone(self,data):
        max_zone = max(data['crowd_data']['zones'])
        '''max_zone is in the form of tuple (zone,crowd_nb,hour)'''
        return max_zone
    
    def get_min_zone(self,data):
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

        ''' END OF ALL JSON RELATED DATA  ''' 
        ######################################
        #              CSV MODE:             #        
        #               BEGINS               #
        ######################################

    def read_day_data(self,csv_file,day):
        time = []
        crowd_number = []
        all_day_info = []
        with open(csv_file, 'r') as file:
            current = csv.DictReader(file)
            for row in current:
                    if dict(row).get('day') == day:
                        time.append(dict(row).get('heure'))
                        crowd_number.append(dict(row).get('count'))
                        all_day_info.append(dict(row))
                        
        time = np.array(time).astype(np.float)
        crowd_number = np.array(crowd_number).astype(np.float)                    
        return all_day_info,time,crowd_number                   
    
    def get_min_hour(self,crowd_number,time):
            min_crowd = min(crowd_number)
            hours = []
            min_index = []
            i = 0
            print(min_crowd)
            for hour in time:
                if min_crowd == crowd_number[i]:
                    hours.append(hour)
                    min_index.append(i)
                i = i + 1    
            return hours, min_index
        
    def get_max_hour(self,crowd_number,time):
            max_crowd = max(crowd_number)
            hours = []
            max_index = []
            i = 0
            for hour in time:
                if max_crowd == crowd_number[i]:
                   hours.append(hour)
                   max_index.append(i)
                i = i + 1   
            return hours,max_index
        
    def get_max_zones(self,day_infos,max_index):
        zones = []
        max_z = 0
        zone_name = ''
        for index in max_index:
            if int(day_infos[index]['Z1']) > int(day_infos[index]['Z2']):
                max_z = int(day_infos[index]['Z1'])
                zone_name = 'Z1'
            else:
                max_z = int(day_infos[index]['Z2'])
                zone_name = 'Z2'
            if max_z < int(day_infos[index]['Z3']) :
                zone_name = 'Z3'
                max_z = int(day_infos[index]['Z3'])
            zones.append( (day_infos[index]['heure'], zone_name, max_z) )    
        
        return zones

    def get_min_zones(self,day_infos,min_index):
        zones = []
        min_z = 0
        zone_name = ''
        for index in min_index:
            if int(day_infos[index]['Z1']) < int(day_infos[index]['Z2']):
                min_z = int(day_infos[index]['Z1'])
                zone_name = 'Z1'
            else:
                min_z = int(day_infos[index]['Z2'])
                zone_name = 'Z2'
            if min_z > int(day_infos[index]['Z3']) :
                zone_name = 'Z3'
                min_z = int(day_infos[index]['Z3'])
            zones.append( (day_infos[index]['heure'], zone_name, min_z) )    
        
        return zones

    def total_crowd_in_zone(self,day_infos):
        total_crowd_zone = [0,0,0]
        zones = ['Z1','Z2','Z3']
        for hour in day_infos:
            total_crowd_zone[0] = total_crowd_zone[0] + int(hour['Z1'])
            total_crowd_zone[1] = total_crowd_zone[1] + int(hour['Z2'])
            total_crowd_zone[2] = total_crowd_zone[2] + int(hour['Z3'])
        return zones, total_crowd_zone    
        
    def initialize(self,app):

        ''' Read csv file '''
        day_infos, time_csv, crowd_number_csv = self.read_day_data(self.data_file,"Dimanche")
        min_hours, min_index = self.get_min_hour(crowd_number_csv,time_csv)
        max_hours, max_index = self.get_max_hour(crowd_number_csv,time_csv)
        zones_maximal = self.get_max_zones(day_infos,max_index)
        zones_minimal = self.get_min_zones(day_infos,min_index)
        zones_names, total_crowd_par_zone = self.total_crowd_in_zone(day_infos)

        '''Generated Pie Chart'''

        #df = px.data.tips()
        #fig = px.pie(df, values='tip', names='day')
        crd_frame = pd.read_csv(self.data_file, delimiter = ',', names=['day','heure','count','Z1','Z2','Z3'])
        fig = px.pie(crd_frame[['day','count']], values='count', names='day',color_discrete_sequence=px.colors.sequential.RdBu,title='Test')
        '''Opening JSON file
        with open(self.data_file, 'r') as openfile: 
            data_points = json.load(openfile)'''

        '''Creating some Cards'''
        #max_crowd = max(data_points['crowd_data']['crowd_number'])
        #min_crowd = min(data_points['crowd_data']['crowd_number'])
        #first_card = dbc.Card( reusable.params_card(self.get_max_hour(data_points),max_crowd,"Heure Max","Heure(s): ","Nombre de personnes: "))
    
        #second_card = dbc.Card(reusable.params_card(self.get_min_hour(data_points),min_crowd,"Heure Min","Heure(s): ","Nombre de personnes: "), color="primary", inverse=True)
        max_crowd = max(crowd_number_csv)
        min_crowd = min(crowd_number_csv)
        first_card = dbc.Card( reusable.params_card(max_hours,max_crowd,"Heure-Max","Heure(s): ","Nombre de personnes dans la scène: "))
    
        second_card = dbc.Card(reusable.params_card(min_hours,min_crowd,"Heure-Min ","Heure(s): ","Nombre de personnes dans la scène: "), color="primary", inverse=True)
        first_card11 = dbc.Card( reusable.params_card_test(zones_maximal,max_crowd,"Heure-Max","Heure(s): ","Nombre de personnes dans la scène: "))

        #zone = [('11', 'Z2', 5),('15', 'Z3', 2),('19.3', 'Z2', 2),('23.45', 'Z1', 2)]
        #second_card1 = dbc.Card(reusable.params_card_test(zones_minimal,min_crowd,"Zone-Min ",zone,"Heure(s): ","Nombre de personnes par zone: "), color="primary", inverse=True)

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
                        #{'x': data_points['crowd_data']['time'], 'y': data_points['crowd_data']['crowd_number'], 'type': 'line', 'name': 'Nombre de foule'}
                        {'x': time_csv, 'y': crowd_number_csv, 'type': 'line', 'name': 'Nombre de foule'}
                    ],
                    'layout': {
                        'title': 'Nombre de la foule dans les différentes heures'
                    }
                }
            )
        ])
        #zones , total_crowd = self.count_crowd_zone(data_points)    
        BarChart = html.Div(children=[
            html.H1(children='Bar Chart'),

            html.Div(children='''
                Nombre total de la foule passé par la zone 'X' dans la journée.
            '''),

            dcc.Graph(
                id='example-graph',
                figure={
                    'data': [
                        {'x': zones_names, 'y': total_crowd_par_zone, 'type': 'bar', 'name': 'SF'},
                    ],
                    'layout': {
                        'title': 'Nombre total de la foule passé par la zone \'X\''
                    }
                }
            )
        ])    
            
        BarChart2 = html.Div(children=[
            html.H1(children='Pie Chart'),

            html.Div(children='''
                Pourcentage de nombre de personnes dans une journée par rapport à la semaine.
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
                        dbc.Col(html.Div(first_card), width=3),
                        dbc.Col(html.Div(second_card), width=3),
                        dbc.Col(html.Div(
                                dbc.Card(dbc.CardBody(
                                    [html.H1("Zone-Max", className="card-title"),
                                     html.P(children=[html.B("Heure(s):"),"8.0, 12.0, 13.0, 14.15, 17.0"], className="h3"),
                                     html.P(children=[html.B("Nombre de personnes par zone:"),"(Z1, 74), (Z2, 96), (Z2, 82), (Z3, 89), (Z1, 98)"], className="h3")   
                                    ]
                                    ))
                            ), width=3),
                        dbc.Col(html.Div(
                                dbc.Card(dbc.CardBody(
                                    [html.H1("Zone-Min", className="card-title"),
                                     html.P(children=[html.B("Heure(s):"),"11.0, 15.0, 19.3, 23.45. "], className="h3"),
                                     html.P(children=[html.B("Nombre de personnes par zone: "),"(Z2,5), ('Z3, 2), (Z2,2), (Z1,2)"], className="h3")   
                                    ]
                                    ),color="primary", inverse=True)
                            ), width=3),
                    ],
                    align="start"
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
                        dbc.Col(html.Div("One of three columns"), align="start", style={'display':'none'}),
                        dbc.Col(html.Div("One of three columns"), align="center", style={'display':'none'}),
                        dbc.Col(html.Div("One of three columns"), align="end", style={'visibility':'hidden'}),
                    ]
                ),
            ]
        )           
    
