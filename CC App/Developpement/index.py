import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
# import grasia_dash_components as gdc
# import dash_defer_js_import as dji
from dash.dependencies import Input, Output

from flask import request
import urllib.parse as urlparse
from urllib.parse import parse_qs
    # User's modules
import config
import languages
from app import app
import components


    # Translation dict


app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname'),Input('url', 'search')])
def display_page(pathname,search):
    config.LANGUAGE_DICT=languages.translate_dict  

    if pathname == '/global':
        component = components.Global_Charts(app, 'ressources/crowd_records.csv')
        return dbc.Container([html.Hr(), component.layout],style={"display": "flex", "flex-direction": "column"}, fluid=True)
    elif pathname == '/view':
        config.LANGUAGE_DICT=dict((key,key) for key in list(config.LANGUAGE_DICT))

        component = components.View(app, config)
        # app.callback_map = component.app.callback_map.copy()
        
        # import_js=dji.Import(src="test.js")
        return dbc.Container([component.layout], fluid=True)
    elif pathname == '/fr/view':
        component = components.View(app, config)
        
        return dbc.Container([component.layout], fluid=True)
    elif pathname == '/statistics':
        config.LANGUAGE_DICT=dict((key,key) for key in list(config.LANGUAGE_DICT))

        
       
        parsed = urlparse.urlparse(search)
        args=parse_qs(parsed.query)
        print(args)

        component = components.StatsView(app, config,args)
        
        return dbc.Container([component.layout], fluid=True)
    elif pathname == '/fr/statistics':

        parsed = urlparse.urlparse(search)
        args=parse_qs(parsed.query)
        print(args)

        component = components.StatsView(app, config,args)
        
        return dbc.Container([component.layout], fluid=True)
    
    elif pathname == '/sensorprocess':
        
        config.LANGUAGE_DICT=dict((key,key) for key in list(config.LANGUAGE_DICT))
        parsed = urlparse.urlparse(search)
        args=parse_qs(parsed.query)
        print(args)

        component = components.SensorProcessView(app, config,args)
        
        return dbc.Container([component.layout], fluid=True)
    
    elif pathname == '/fr/sensorprocess':
        
       
        parsed = urlparse.urlparse(search)
        args=parse_qs(parsed.query)
        print(args)

        component = components.SensorProcessView(app, config,args)
        
        return dbc.Container([component.layout], fluid=True)
    elif pathname == '/sensors_dashboard':
        
        config.LANGUAGE_DICT=dict((key,key) for key in list(config.LANGUAGE_DICT))
        parsed = urlparse.urlparse(search)
        args=parse_qs(parsed.query)

        component = components.SensorsDashboardView(app, config,args)
        
        return dbc.Container([component.layout], fluid=True)
    
    elif pathname == '/fr/sensors_dashboard':
        
       
        parsed = urlparse.urlparse(search)
        args=parse_qs(parsed.query)

        component = components.SensorsDashboardView(app, config,args)
        
        return dbc.Container([component.layout], fluid=True)
    else:
        return dbc.Container([dbc.Row([
                                html.H3([html.Span(className='fa fa-ban mr-3'),'404 Page not found'])
                            ],className="h-100 d-flex justify-content-center align-items-center")
                            ],
            fluid=True,style={'height':'100vh'})


if __name__ == '__main__':
    default_callbacks = app.callback_map
    app.run_server(debug=True)
