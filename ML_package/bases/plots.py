import os,sys,glob,inspect
import plotly.graph_objects as go
import numpy as np
import random

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
    # User's modules from another directory
sys.path.append(os.path.join(parentdir, "bases"))
import utils
from utils import *
COLOR_PALETTE = [(255, 23, 68), (240, 98, 146), (170, 0, 255), (124, 67, 189), (48, 79, 254), (26, 35, 126), (41, 121, 255),
                 (0, 145, 234), (24, 255, 255), (0, 230,
                                                 118), (50, 203, 0), (0, 200, 83), (255, 255, 0),
                 (255, 111, 0), (172, 25, 0), (84, 110, 122), (213, 0, 0), (250, 27, 27)]


def showLineChart(list_axes: list, names: list, title=None, x_title=None, y_title=None, special_points=[]):

    fig = go.FigureWidget()
    if len(list_axes) != len(names):
        raise Exception("Names list length doesn't match axises list length.")
    colors_set = random.sample(COLOR_PALETTE, len(list_axes))
    for i, (x, y) in enumerate(list_axes):
        fig.add_trace(go.Scatter(x=x, y=y, name=names[i], line=dict(
            color='rgb'+str(colors_set[i]))))
    fig.update_layout(title={'text': title if title is not None else '',
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'
                             },
                      xaxis_title=x_title if x_title is not None else '',
                      yaxis_title=y_title if y_title is not None else '',
                      font=dict(
                          family="Courier New, monospace",
                          size=18,
                          color="#7f7f7f"
                        ),
                    annotations=[
                    dict(
                        x=pt[0],
                        y=pt[1],
                        xref="x",
                        yref="y",
                        text=pt[2],
                        ax=0,
                        ay=-40
                    ) for pt in special_points
                ]
    )
    fig.show()
    return fig


if __name__ == "__main__":
    x1 = [2, 4, 6, 8, 9, 11, 15, 19]
    x2 = [3, 4, 6, 7, 10, 12, 16, 20]
    x3 = [2, 4, 7, 8, 9, 11, 15, 22]
    x4 = [2, 4, 6, 8, 9, 11, 15, 19]
    y1 = [2.12, 4.12, 6.12, 8.12, 9.12, 11.12, 15.12, 19]
    y2 = [30.1, 4.8, 1.8, 12.8, 4.8, 22.8, 20.8, 19.2]
    y3 = [6.5, 1.5, 32.5, 4.5, 5.5, 22.5, 10.5, 19]
    y4 = [22.12, 14.12, 46.12, 38.12, 9.12, 13.2, 15.2, 19]
    names = ['D1', 'D6', 'D5', 'D4']
    liste = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    

    showLineChart(liste, names, title="Line plot",special_points=[(2,2.12,'min error'),(19,19,'minimum_error')])
