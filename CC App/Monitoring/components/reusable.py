
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

def params_card(hours,crowd_number,param_name,title1,title2):
    card_content = [html.H1(param_name, className="card-title")]
    string = ""
    i = 1
    for hour in hours:
        if i < len(hours):
            string = string+str(hour)+", "
        string = string+str(hour)+"."
    card_content.append(html.P(children=[html.B(title1),string],className="h3")) 
    card_content.append(html.P(children=[html.B(title2),str(crowd_number)],className="h3"))
    return dbc.CardBody(card_content)     
        
    