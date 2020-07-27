
import dash
import dash_bootstrap_components as dbc
from flask import Flask, Response,send_from_directory,request,jsonify
import multiprocessing
import os
# User's modules
import config
from modelmanager import ModelManager

# scene_region_params=None
scene_region_params={
    'tang': 1.9528301886792452,
    'b': -272
}

STATIC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ressources','assets')
# external JavaScript files
external_scripts = [
    # {
    #     'src': 'https://code.jquery.com/jquery-3.4.1.slim.min.js',
    #     'integrity': 'sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n',
    #     'crossorigin': 'anonymous'
    # },
    # {
    #     'src': '"https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js',
    #     'integrity': 'sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo',
    #     'crossorigin': 'anonymous'
    # },
    # {
    #     'src': 'https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js',
    #     'integrity': 'sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6',
    #     'crossorigin': 'anonymous'
    # },
    # {
    #     'src': 'assets/js/view-custom.js'

    # }
    {
        'src': 'http://localhost:8050/assets/js***jquery-3.4.1.min.js',
       
    },
    {
        'src': '"http://localhost:8050/assets/js***popper.min.js',
       
    },
    {
        'src': 'http://localhost:8050/assets/js***bootstrap-4.4.1.min.js',
     
    },
    {
        'src': 'http://localhost:8050/assets/js***attrchange.js',
     
    },
    {
        'src': 'http://localhost:8050/assets/css***fontawesome***js***all.js',
        
       
    },
    {
        'src': 'http://localhost:8050/assets/js***socketio.js',
     
    },
    {
        'src': 'http://localhost:8050/assets/js***view-custom.js',
        'type': 'application/javascript'

    }

]

# external CSS stylesheets
external_stylesheets = [

    dbc.themes.BOOTSTRAP,
   
    {
        'href': 'http://localhost:8050/assets/css***fontawesome***css***all.css',
        'rel': 'stylesheet',
       
    },
    # {
    #     'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css',
    #     'rel': 'stylesheet',
    #     'integrity': 'sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh',
    #     'crossorigin': 'anonymous'
    # },
    # {
    #     'href': 'https://use.fontawesome.com/releases/v5.7.0/css/all.css',
    #     'rel': 'stylesheet',
    #     'integrity': 'sha384-lZN37f5QGtY3VHgisS14W3ExzMWZxybE1SJSEsQp9S+oqd12jhcu+A56Ebc1zFSJ',
    #     'crossorigin': 'anonymous'
    # },
    # {
    #     'href': "https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.0.0/animate.min.css",
    #     'rel': "stylesheet"
    # },
    {
        'href': "http://localhost:8050/assets/css***animate-css.css",
        'rel': "stylesheet"
    },
    {
        'href': "http://localhost:8050/assets/css***fonts.css",
        'rel': "stylesheet"
    },
    {
        'href': "http://localhost:8050/assets/css***internal.css",
        'rel': "stylesheet"
    }
    # {
    #     'href': "http://localhost:8050/assets/css***base.css",
    #     'rel': "stylesheet"
    # },
    # 'https://codepen.io/chriddyp/pen/bWLwgP.css',
    
]

server=Flask(__name__)
app = dash.Dash(__name__,
                server=server,
                external_scripts = external_scripts,
                external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)

@server.route("/assets/<path>")
def serve_file(path):
    fullpath=path.split('***')
    return send_from_directory(os.path.sep.join([STATIC_PATH]+fullpath[:-1]), fullpath[-1])

@server.route("/scene/regions/",methods=['POST'])
def save_regions_params():
    global scene_region_params
    if scene_region_params is None:
        scene_region_params=dict()
    params=request.values
    scene_region_params['tang']=params['tang']
    scene_region_params['b']=params['b']
    return {
        'status':'ok',
        'statuscode':200
    }
    
       

# server = app.server
# server.run(port=4000)
app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions'] = True


ModelManager.set_base_path(config.FROZEN_MODELS_BASE_PATH)
def get_regions_params():
    global scene_region_params
    return scene_region_params
# Running the server
if __name__ == '__main__':

    app.run_server(dev_tools_hot_reload=False,
                   debug=config.DEBUG, host='0.0.0.0')

    # model=ModelManager.load_external_model('CSRNet')
