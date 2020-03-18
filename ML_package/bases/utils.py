import os, glob,sys,inspect

    # Root path of the project
BASE_PATH=os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))


def path_exists(path):
    path=[el for el in path.split('/') if el!='.']
    
    return os.path.exists(os.path.join(BASE_PATH,*path))