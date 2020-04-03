import os, glob,sys,inspect

    # Root path of the ML Package
BASE_PATH=os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))


def path_exists(path):
    path=[el for el in path.split('/') if el!='.']
    
    return os.path.exists(os.path.join(BASE_PATH,*path))

def make_path(dir_path):
    if not os.path.exists(dir_path):
        make_path(os.path.split(dir_path)[0])
        os.mkdir(dir_path)

def split_path(path):
     ( head, tail ) = os.path.split(path)
     return split_path(head) + [ tail ] if head and head != path else [ head or tail ]


