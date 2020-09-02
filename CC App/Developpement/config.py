import os,inspect
from PIL import Image, ImageDraw, ImageFont
import numpy as np

FROZEN_MODELS_BASE_PATH=os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))),'store','models')
VIDEOS_DIR_PATH=os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))),'ressources','videos')
SENSORS_DEFAULT_BASE_PATH=os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))),'store','sensors')
DEBUG = True
FRAMERATE = 24.0
LANGUAGE_DICT=None


def print_ascii_large(text, font_size=18):

    myfont = ImageFont.truetype("verdanab.ttf", font_size)
    img = Image.new("1", myfont.getsize(text), "black")
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, "white", font=myfont)
    pixels = np.array(img, dtype=np.uint8)
    chars = np.array([' ', '#'], dtype="U1")[pixels]
    strings = chars.view('U' + str(chars.shape[1])).flatten()
    print()
    for s in strings:
        if len(s.strip()) > 0:
            print(s)
    print()
'''
================= DB.JSON FAKE SERVER AU CAS OÙ YE7BASS DO THIS COMMANDS : =========================

echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf && sudo sysctl -p

cat /proc/sys/fs/inotify/max_user_watches

fs.inotify.max_user_watches=524288

======
'''
