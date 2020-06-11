import os,inspect
FROZEN_MODELS_BASE_PATH=os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))),'store','models')
VIDEOS_DIR_PATH=os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))),'ressources','videos')
DEBUG = True
FRAMERATE = 24.0