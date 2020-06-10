import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import grasia_dash_components as gdc
import dash_defer_js_import as dji
from dash.dependencies import Input, Output

from app import app
import components

app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):

    if pathname == '/global':
        # app.callback_map = default_callbacks
        component = components.Global_Charts(app, 'ressources/dummy_data.json')
        return dbc.Container([html.Hr(), component.layout],style={"display": "flex", "flex-direction": "column"}, fluid=True)
    elif pathname == '/mono':
        # app.callback_map = default_callbacks
        component = components.Mono_Chart(app)
        return dbc.Container([component.layout], fluid=True, style={"height": "500px", })
    elif pathname == '/view':
        # app.callback_map = default_callbacks
        component = components.View(app, None)
        # app.callback_map = component.app.callback_map.copy()
        print('getting')
        
        import_js=dji.Import(src="test.js")
        return dbc.Container([component.layout,import_js], fluid=True, style={"height": "500px", })
    else:
        return dbc.Container([dbc.Row([
                                html.H3([html.Span(className='fa fa-ban mr-3'),'404 Page not found'])
                            ],className="h-100 d-flex justify-content-center align-items-center")
                            ],
            fluid=True,style={'height':'100vh'})


if __name__ == '__main__':
    default_callbacks = app.callback_map
    app.run_server(debug=True)
