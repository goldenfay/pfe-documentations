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
    sensors_dirs=[dirname for dirname in os.listdir(self.sensor_path) if os.path.isdir(os.path.join(self.sensor_path,dirname))]

    for sdir in sensors_dirs:
        try:
            with open(os.path.join(sdir,'temp.csv')) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                json[sdir]=[]
        except:
            pass        


def read_existing_data(filename)->pd.DataFrame:
    times = []
    values = []
    if os.path.isfile(filename):
        with open(filename) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                times.append(datetime.datetime.strptime(row[0], "%Y%m%d_%H-%M-%S "))
                values.append(int(row[1]))
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

    # plt.gcf().clear()

    # plt.subplot(2, 2, 1)
    # plt.plot(df_1w.index.tolist(), df_1w['value'].tolist())
    # plt.title("Laatste week")
    # plt.ylabel("Personen")
    # plt.xlabel("Tijdstip")

    # plt.subplot(2, 2, 2)
    # plt.plot(df_1d.index.tolist(), df_1d['value'].tolist())
    # plt.title("Afgelopen 24 uur")
    # plt.ylabel("Personen")
    # plt.xlabel("Tijdstip")

    # plt.subplot(2, 2, 3)
    # plt.plot(df_8h.index.tolist(), df_8h['value'].tolist())
    # plt.title("Afgelopen 8 uur")
    # plt.ylabel("Personen")
    # plt.xlabel("Tijdstip")

    # plt.subplot(2, 2, 4)
    # plt.plot(df_2h.index.tolist(), df_2h['value'].tolist())
    # plt.title("Afgelopen 2 uur")
    # plt.ylabel("Personen")
    # plt.xlabel("Tijdstip")

    # plt.gcf().autofmt_xdate()
    if standalone_window:
        plt.show()
    else:
        # fig= plt_tools.make_subplots(rows=3, cols=1, shared_xaxes=False,vertical_spacing=0.009,horizontal_spacing=0.009)
        # fig.append_trace({'x':df_1w.index,'y':df_1w['value'].values.tolist(),'type':'scatter','name':'Week'},1,1)
        # fig.append_trace({'x':df_1d.index,'y':df_1d['value'].values.tolist(),'type':'scatter','name':'Day'},2,1)
        # fig.append_trace({'x':df_2h.index,'y':df_2h['value'].values.tolist(),'type':'scatter','name':'Hours'},3,1)
        # return plt_tools.mpl_to_plotly(plt.gcf())    
        return [df_2h,df_8h,df_1d,df_1w]   
        # return fig 




def index_to_list_date(timeIndex):
    return ['{}/{}/{}'.format(timestamp.year,timestamp.month,timestamp.day) for timestamp in timeIndex]

dataframe = pd.DataFrame()
dataframe['timestamp'] = pd.Series(dtype='datetime64[ns]')
dataframe['value'] = pd.Series(dtype=np.int32)   
for i in range(10):
    dataframe=dataframe.append({'timestamp': pd.Timestamp(datetime.datetime.now()),'value':i},ignore_index=True) 
dataframe.set_index('timestamp', inplace=True)
print(dataframe)    