from textwrap import dedent
import dash
import dash_bootstrap_components as dbc
import config
import re
import time
import traceback

# User's modules
from modelmanager import ModelManager


# external JavaScript files
external_scripts = [

    {
        'src': 'https://code.jquery.com/jquery-3.4.1.slim.min.js',
        'integrity': 'sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n',
        'crossorigin': 'anonymous'
    },
    {
        'src': '"https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js',
        'integrity': 'sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo',
        'crossorigin': 'anonymous'
    },
    {
        'src': 'https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js',
        'integrity': 'sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6',
        'crossorigin': 'anonymous'
    },
    {
        'src': 'assets/js/view-custom.js'

    }

]

# external CSS stylesheets
external_stylesheets = [

    {
        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh',
        'crossorigin': 'anonymous'
    },
    {
        'href': 'https://use.fontawesome.com/releases/v5.7.0/css/all.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-lZN37f5QGtY3VHgisS14W3ExzMWZxybE1SJSEsQp9S+oqd12jhcu+A56Ebc1zFSJ',
        'crossorigin': 'anonymous'
    },
    {
        'href': "https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.0.0/animate.min.css",
        'rel': "stylesheet"
    },
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    dbc.themes.BOOTSTRAP
]
app = dash.Dash(__name__,
                external_scripts == external_scripts,
                external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)
server = app.server

app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions'] = True


ModelManager.set_base_path(config.FROZEN_MODELS_BASE_PATH)
# Running the server
if __name__ == '__main__':

    app.run_server(dev_tools_hot_reload=False,
                   debug=config.DEBUG, host='0.0.0.0')

    # model=ModelManager.load_external_model('CSRNet')
