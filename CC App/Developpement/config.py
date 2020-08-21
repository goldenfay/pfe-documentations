import os,inspect
FROZEN_MODELS_BASE_PATH=os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))),'store','models')
VIDEOS_DIR_PATH=os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))),'ressources','videos')
SENSORS_DEFAULT_BASE_PATH=os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))),'store','sensors')
DEBUG = True
FRAMERATE = 24.0
LANGUAGE_DICT=None


'''
================= DB.JSON FAKE SERVER AU CAS OÙ YE7BASS DO THIS COMMANDS : =========================

echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf && sudo sysctl -p

cat /proc/sys/fs/inotify/max_user_watches

fs.inotify.max_user_watches=524288

======
'''
