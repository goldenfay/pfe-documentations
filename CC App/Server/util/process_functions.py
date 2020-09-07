import numpy as np
import pandas as pd
import plotly.graph_objs as go
from matplotlib import cm
from io import BytesIO as _BytesIO
from PIL import Image
import cv2,imutils
import base64

def b64_to_pil(string):
    decoded = base64.b64decode(string)
    buffer = _BytesIO(decoded)
    im = Image.open(buffer)

    return im


def b64_to_numpy(string, to_scalar=True):
    im = b64_to_pil(string)
    np_array = np.asarray(im)[:, :, :3]

    if to_scalar:
        np_array = np_array / 255.

    return np_array


def numpy_to_pil(array, jetMap=True):
    if jetMap:
        print('\t Converting to Jet color map')
        array = cm.jet(array)
    return Image.fromarray(np.uint8(array*255))


def numpy_to_b64(array, jetMap=True):
    im_pil = numpy_to_pil(array, jetMap)
    buff = _BytesIO()
    im_pil.save(buff, format="png")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def pil_to_b64(im, enc_format='png', verbose=False, **kwargs):

    buff = _BytesIO()
    im.save(buff, format=enc_format, **kwargs)
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")
    return encoded


def concat_frame_dmap(frame,dmap):
    min_height=min(dmap.shape[0],frame.shape[0])
    frame=imutils.resize(frame, height=min_height)
    dmap = imutils.resize(dmap, height=min_height)
    dmap=cm.jet(dmap)*255
    dmap=dmap[:,:,:3].astype('uint8')
    dmap=cv2.cvtColor(dmap, cv2.COLOR_BGR2RGB)
    nb_rows= int(dmap.shape[0]+frame.shape[0])
    nb_cols= int(dmap.shape[1]+frame.shape[1])
    # concated = np.zeros(shape=(min_height, nb_cols, 3), dtype=np.uint8)
    # concated[:,:frame.shape[1]]=frame
    # concated[:,frame.shape[1]:]=dmap[:,:,:3]
    frame=frame*255
    concated=cv2.hconcat([frame.astype('uint8'),dmap[:,:,:3]])
    return concated