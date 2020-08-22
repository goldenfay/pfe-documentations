import numpy as np
import pandas as pd
import csv
from matplotlib import cm
import matplotlib.pyplot as plt
import plotly.tools as plt_tools
from io import BytesIO as _BytesIO
from PIL import Image
import os,base64,datetime


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


def log_count(filename, n):
  
    f = open(filename, "a")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    line = "{} , {}\n".format(timestamp, n)
    f.write(line)
    f.close()

def construct_combined_results(dirpath):
    json={}
    sensors_dirs=[dirname for dirname in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath,dirname))]

    for sdir in sensors_dirs:
        try:
            if os.path.exists(os.path.join(dirpath,sdir,'output','temp.csv')):
                json[sdir]=read_existing_data(os.path.join(dirpath,sdir,'output','temp.csv'))
            
        except:
            pass 

    return json           


def read_existing_data(filename)->pd.DataFrame:
    times = []
    values = []
    if os.path.isfile(filename):
        with open(filename) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                times.append(datetime.datetime.strptime(row[0], "%Y%m%d_%H-%M-%S "))
                values.append(int(float(row[1])))
    dataframe = pd.DataFrame()
    dataframe['timestamp'] = pd.Series(dtype='datetime64[ns]')
    dataframe['value'] = pd.Series(dtype=np.int32)
    dataframe['timestamp'] = times
    dataframe['value'] = values
    dataframe.set_index('timestamp', inplace=True)
    return dataframe

def show_plots(data:pd.DataFrame,standalone_window=False):
    """
    Show the graphs with historical data
    :param data: dataframe
    :return:
    """
    
    # data.index = pd.to_datetime(data.index)
    data.index = pd.DatetimeIndex(data.index)
    # Awful code to create new dataframes each time the graph is shown
    df_1w = data[data.index >= pd.datetime.now() - pd.Timedelta('7D')]
    df_1d = df_1w[df_1w.index >= pd.datetime.now() - pd.Timedelta('24H')]
    df_8h = df_1d[df_1d.index >= pd.datetime.now() - pd.Timedelta('8H')]
    df_2h = df_8h[df_8h.index >= pd.datetime.now() - pd.Timedelta('2H')]
    # Resample to smooth the long running graphs
    df_1w = df_1w.resample('1H').max()
    df_1d = df_1d.resample('15min').max()

    
    if standalone_window:
        plt.show()
    else:
        
        return [df_2h,df_8h,df_1d,df_1w]   




def index_to_list_date(timeIndex):
    return ['{}/{}/{}'.format(timestamp.year,timestamp.month,timestamp.day) for timestamp in timeIndex]

dataframe = pd.DataFrame()
dataframe['timestamp'] = pd.Series(dtype='datetime64[ns]')
dataframe['value'] = pd.Series(dtype=np.int32)   
for i in range(10):
    dataframe=dataframe.append({'timestamp': pd.Timestamp(datetime.datetime.now()),'value':i},ignore_index=True) 
dataframe.set_index('timestamp', inplace=True)
