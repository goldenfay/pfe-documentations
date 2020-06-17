
import dash
import dash_bootstrap_components as dbc
from flask import Flask, Response
import multiprocessing
import queue
# User's modules
import config
from modelmanager import ModelManager


# QUEUE=multiprocessing.Queue()
QUEUE=queue.Queue()
seekindex=0


# external JavaScript files
external_scripts = [

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

server=Flask(__name__)
app = dash.Dash(__name__,
                server=server,
                external_scripts = external_scripts,
                external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)
@server.route("/video_feed")
def feed_video():
    global seekindex
    # global QUEUE
    
    # if not QUEUE.empty():
    #     return Response(QUEUE.get_nowait(),mimetype='multipart/x-mixed-replace; boundary=frame')
    # else:
    #     return Response('Loading') 
    with open('temp.txt','r') as f:
        lines=f.readlines()
        if len(lines)>seekindex   :
            res= Response(lines[seekindex],mimetype='multipart/x-mixed-replace; boundary=frame')
            seekindex+=1
            return res
        else:
            return Response('Loading')     

# server = app.server
# server.run(port=4000)
app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions'] = True


ModelManager.set_base_path(config.FROZEN_MODELS_BASE_PATH)
# Running the server
if __name__ == '__main__':

    app.run_server(dev_tools_hot_reload=False,
                   debug=config.DEBUG, host='0.0.0.0')

    # model=ModelManager.load_external_model('CSRNet')
