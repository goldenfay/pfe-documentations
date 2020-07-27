
import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import dash_bootstrap_components as dbc

def statistical_card(title, Value, id=None, css_class=None):
    return dbc.Card()


def dropdown_control(label, dropdown_options, value, **kargs):
    return html.Div(
        className='control-element',
        children=[
            html.Div(children=[label], style={
                'width': '40%'}),
            dcc.Dropdown(
                options=dropdown_options,
                value=value,
                searchable=False,
                clearable=False,
                style={'width': '60%'},
                **kargs
            )
        ]
    )


def drag_drop_container(id, children_id, dragdrop_labels):
    return html.Div(
        children=[dcc.Upload(
            id=id,
            className='d-flex align-items-center justify-content-center',
            children=html.Div(
                id=children_id,
                className='align-self-center',
                children=[
                    html.Div([dragdrop_labels[0]]),
                    html.Div([dragdrop_labels[1]]),
                    html.Div([html.A(dragdrop_labels[2])])
                ]),
            style={
                'width': '500px',
                'height': '300px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                                'textAlign': 'center',
                                'fontWeight': 'bold'

            },

            # Allow multiple files to be uploaded
            multiple=True
        )]
    )


def params_card(hours, crowd_number, param_name, title1, title2):
    card_content = [html.H1(param_name, className="card-title")]
    string = ""
    i = 1
    for hour in hours:
        if i < len(hours):
            string = string+str(hour)+", "
        string = string+str(hour)+"."
    card_content.append(
        html.P(children=[html.B(title1), string], className="h3"))
    card_content.append(
        html.P(children=[html.B(title2), str(crowd_number)], className="h3"))
    return dbc.CardBody(card_content)


def count_results_grid(images_list,res_img_list):
    return [
        html.Div(className='row mt-5', children=[

            html.Div(children=[
                html.Div(html.H4('Original', className='muted'),
                         className="d-flex justify-content-center"),
                html.Img(id='img-org-{}'.format(id),
                         src=images_list[i], style={
                    'width': '100%'})

            ],
                className='col-md justify-content-center animate__animated animate__fadeInRight'),
            html.Div(children=[
                html.Div(html.H4('Estimated count : '+str(int(count)),
                                 className='muted'), className="d-flex justify-content-center"),
                html.Img(id='img-{}'.format(id),
                         src=encoded_img, style={
                    'width': '100%'})

            ],
                className='col-md justify-content-center animate__animated animate__fadeInRight')
        ])
        for (i, (id, encoded_img, count)) in enumerate(res_img_list)]
