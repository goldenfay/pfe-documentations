import dash
import dash_bootstrap_components as dbc


'''BOOTSTRAP DEPENDENCIES'''    
external_stylesheets =['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]
'''Instanciation'''

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = 'Graphics UI'
server = app.server

