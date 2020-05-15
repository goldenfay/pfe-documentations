import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc


BarChart = html.Div(children=[
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

layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(html.Div(BarChart)),
            ],
            align="start",
        )
    ]
)
