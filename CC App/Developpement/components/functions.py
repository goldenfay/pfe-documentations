import numpy as np
import pandas as pd
from matplotlib import cm
from io import BytesIO as _BytesIO
from PIL import Image
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
    print(array.shape,array.max(),array.min())    
    if jetMap:
        print('\t Converting to Jet color map')
        array = cm.jet(array)
    array=array/array.max()*255    
    return Image.fromarray(np.uint8(array))


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


def load_data(path):

    # Load the dataframe containing all the processed object detections inside the video
    video_info_df = pd.read_csv(path)

    # The list of classes, and the number of classes
    classes_list = video_info_df["class_str"].value_counts().index.tolist()
    n_classes = len(classes_list)

    # Gets the smallest value needed to add to the end of the classes list to get a square matrix
    root_round = np.ceil(np.sqrt(len(classes_list)))
    total_size = root_round ** 2
    padding_value = int(total_size - n_classes)
    classes_padded = np.pad(classes_list, (0, padding_value), mode='constant')

    # The padded matrix containing all the classes inside a matrix
    classes_matrix = np.reshape(
        classes_padded, (int(root_round), int(root_round)))

    # Flip it for better looks
    classes_matrix = np.flip(classes_matrix, axis=0)

    data_dict = {
        "video_info_df": video_info_df,
        "n_classes": n_classes,
        "classes_matrix": classes_matrix,
        "classes_padded": classes_padded,
        "root_round": root_round
    }

    # if DEBUG:
    #     print(f'{path} loaded.')

    return data_dict



