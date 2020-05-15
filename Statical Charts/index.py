import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from app import app
import Global_Charts,Mono_Chart

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/global':
        return dbc.Container([html.Hr(),Global_Charts.layout],fluid=True)
    elif pathname == '/mono':
        return dbc.Container([Mono_Chart.layout],fluid=False,style={"height":"500px",})
    else:
        return '404 Page not found'

if __name__ == '__main__':
    app.run_server(debug=True)

