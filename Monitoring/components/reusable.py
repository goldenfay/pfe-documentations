
import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import dash_bootstrap_components as dbc



def statistical_card(title,Value,id=None,css_class=None):
    return dbc.Card()


def dropdown_control(label,dropdown_options,value,**kargs):
    return html.Div(
                                            className='control-element',
                                            children=[
                                                html.Div(children=[label], style={
                                                    'width': '40%'}),
                                                dcc.Dropdown(
                                                    options=dropdown_options ,
                                                    value=value,
                                                    searchable=False,
                                                    clearable=False,
                                                    style={'width': '60%'},
                                                    **kargs
                                                )
                                            ]
                                        )

