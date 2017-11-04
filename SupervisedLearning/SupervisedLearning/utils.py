import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.lines as mlines
import random
from sklearn.model_selection import ParameterGrid
import pandas as pd
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def WriteTextToFile(file, text):
    f = open(file, 'w')
    f.write(text)
    f.close()

def WriteTextArrayToFile(file, textArray):
    WriteTextToFile(file, '\n'.join(textArray))


def ReadLinesFromFile(file):
    f = open(file, 'r')
    lines = [line.rstrip('\n') if(line != '\n')
             else line for line in f.readlines()]
    f.close()
    return lines


def ReadLineFromFile(file, line_idx):
    """
    line_idx starts from 0
    so line number would be line idx + 1
    """
    f = open(file, 'r')
    index = 0
    line_at_idx = None
    for line in f.readlines():
        if(index == line_idx):
            line_at_idx = line
            break
        index = index + 1
    f.close()
    return line_at_idx

def SaveDataPlotWithLegends(YSeries_array,
                 x=None,
                 filename=None,
                 dispose_fig=True,
                 x_axis_name="",
                 y1_axis_name="",
                 title="",
                 legend_loc = 2,
                 y_limits = None,
                 x_limits = None):

    # https://stackoverflow.com/questions/8409095/matplotlib-set-markers-for-individual-points-on-a-line
    fig, ax1 = plt.subplots()
    legends = []
    for y in YSeries_array:
        if(y.xvalues is not None):
            x_values = y.xvalues
        elif(type(x[0]) == str):
            x_values = np.arange(len(x))
            plt.xticks(x_values,x)
        else:
            x_values = x
        ax1.plot(x_values, y.values, linestyle=y.line_style,
                    color=y.line_color, marker=y.points_marker)
        if(y.plot_legend_label is not None):
            legends.append(
                mlines.Line2D([], [],
                                color=y.line_color,
                                linestyle='',
                                marker=y.legend_marker,
                                label=y.plot_legend_label))

    ax1.set_xlabel(x_axis_name)
    ax1.set_ylabel(y1_axis_name)
    if(y_limits is not None):
        ax1.set_ylim(y_limits)
    if(x_limits is not None):
        ax1.set_xlim(x_limits)
    if(len(legends) > 0):
        lgd = plt.legend(handles=legends,bbox_to_anchor=(1, 1),loc=legend_loc)

    if(title != "" or title is not None):
        plt.title(title)

    if filename is not None:
        if(len(legends) > 0):
            fig.savefig(filename,bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            fig.savefig(filename)
        

    if dispose_fig is True:
        plt.close(fig)
        return [None, None]
    else:
        return [fig, ax1]

def SaveScatterPlotWithLegends(YSeries_array,
                 x=None,
                 filename=None,
                 dispose_fig=True,
                 x_axis_name="",
                 y1_axis_name="",
                 title="",
                 legend_loc = 2,
                 y_limits = None,
                 x_limits = None):

    # https://stackoverflow.com/questions/8409095/matplotlib-set-markers-for-individual-points-on-a-line
    fig, ax1 = plt.subplots()
    legends = []
    for y in YSeries_array:
        if(y.xvalues is not None):
            x_values = y.xvalues
        elif(type(x[0]) == str):
            x_values = np.arange(len(x))
            plt.xticks(x_values,x)
        else:
            x_values = x
        ax1.scatter(x_values, y.values, linestyle=y.line_style,
                    color=y.line_color, marker=y.points_marker)
        if(y.plot_legend_label is not None):
            legends.append(
                mlines.Line2D([], [],
                                color=y.line_color,
                                linestyle='',
                                marker=y.legend_marker,
                                label=y.plot_legend_label))

    ax1.set_xlabel(x_axis_name)
    ax1.set_ylabel(y1_axis_name)
    if(y_limits is not None):
        ax1.set_ylim(y_limits)
    if(x_limits is not None):
        ax1.set_xlim(x_limits)
    if(len(legends) > 0):
        lgd = plt.legend(handles=legends,bbox_to_anchor=(1, 1),loc=legend_loc)

    if(title != "" or title is not None):
        plt.title(title)

    if filename is not None:
        if(len(legends) > 0):
            fig.savefig(filename,bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            fig.savefig(filename)
        

    if dispose_fig is True:
        plt.close(fig)
        return [None, None]
    else:
        return [fig, ax1]

def SaveDataPlot(y1,
                 y2=None,
                 x=None,
                 filename=None,
                 dispose_fig=True,
                 x_axis_name="",
                 y1_axis_name="",
                 y2_axis_name="",
                 title="",
                 y1_plot_color='b',
                 y2_plot_color='r',
                 fig=None,
                 ax1=None,
                 ax2=None):

    if(x is None):
        x_count = y1.size
        if(y2 is not None and y2.size > x_count):
            x_count = y2.size
        x = [i + 1 for i in range(x_count)]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(x, y1, y1_plot_color)
    ax1.set_xlabel(x_axis_name)
    ax1.set_ylabel(y1_axis_name, color=y1_plot_color)

    if y2 is not None:
        ax2.plot(x, y2, y2_plot_color)
        ax2.set_ylabel(y2_axis_name, color=y2_plot_color)

    if(title != "" or title is not None):
        plt.title(title)

    if filename is not None:
        fig.savefig(filename)

    if dispose_fig is True:
        plt.close(fig)
    else:
        return [fig, ax1, ax2]


def SaveDataSubplots(
    y1_matrix,
    nrows=1,
    ncols=1,
    y2_matrix=None,
    x_matrix=None,
    filename=None,
    dispose_fig=True,
    x_axis_name="",
    y1_axis_name="",
    y2_axis_name="",
    title="",
    y1_plot_color='b-',
    y2_plot_color='r-',
):
    """
    Creates a grid of size nrows X ncols subplots
    y1[y2] : each column wii be a new time series
    title : list
    """
    assert(y1_matrix.shape[1] == nrows * ncols and (y2_matrix is None or y2_matrix.shape[1]
                                                    == nrows * ncols) and (x_matrix is None or y2_matrix.shape[1] == nrows * ncols))
    fig, axs1 = plt.subplots(nrows, ncols)
    axs1 = [axs1] if nrows == 1 and ncols == 1 else axs1.flatten()
    axs2 = [ax.twinx() for ax in axs1]

    def f(g): return g if isinstance(g, list) else [
        g for i in range(y1_matrix.shape[1])]

    x_axis_name = f(x_axis_name)
    y1_axis_name = f(y1_axis_name)
    y2_axis_name = f(y2_axis_name)
    title = f(title)

    for i in range(y1_matrix.shape[1]):
        y1 = y1_matrix[:, i]
        y2 = None if y2_matrix is None else y2_matrix[:, i]
        ax1 = axs1[i]
        ax2 = axs2[i]

        if(x_matrix is None):
            x_count = y1.size
            if(y2 is not None and y2.size > x_count):
                x_count = y2.size
            x = np.arange(x_count) + 1

        ax1.plot(x, y1, y1_plot_color)
        ax1.set_xlabel(x_axis_name[i])
        ax1.set_ylabel(y1_axis_name[i], color=y1_plot_color)

        if y2 is not None:
            ax2.plot(x, y2, y2_plot_color)
            ax2.set_ylabel(y2_axis_name[i], color=y2_plot_color)

        if(title[i] != ""):
            ax1.set_title(title[i])
            # plt.title(title)

    if filename is not None:
        fig.savefig(filename)

    if dispose_fig is True:
        plt.close(fig)
    else:
        return [fig, axs1, axs2]


def SaveHistogram(data, nbins, title="", x_axis_name="", y_axis_name="", file=None, show=False):
    f = plt.figure()
    plt.hist(data, nbins)
    plt.title(title)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    if(file is not None):
        f.savefig(file)
    if(show):
        plt.show()


def CreateDirectoryIfNotExists(path, is_file=True):
    dir = os.path.dirname(path) if(is_file) else path
    if not os.path.exists(dir):
        os.makedirs(dir)


def PreparePath(path, is_file=True):
    CreateDirectoryIfNotExists(path, is_file)
    return path


def Get_Subdirectories(a_dir):
    l = [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
    l.sort()
    return l

def FilterRows(data, filter_fn):
    return data[data.apply(filter_fn, axis=1)]

def ConcatToStr(delimiter, values_array):
    return delimiter.join([str(x) for x in values_array])

def GetMarkerColorCombinations(seed = 0):
    grid = ParameterGrid([{'marker':['o','x','d','^','+','v','8','s','p','>','<'], 'color':['r','b','g','k','m','y','c']}])
    combinations = [p for p in grid]
    random.seed(seed)
    random.shuffle(combinations)
    return combinations

def GetColorCombinations(seed = 0):
    grid = ParameterGrid([{'color':['orange','red','blue','green','black','saddlebrown','violet','darkcyan','maroon','lightcoral']}])
    combinations = [p for p in grid]
    random.seed(seed)
    random.shuffle(combinations)
    return combinations

def WriteBinaryFile(filename, binobj):
    with open(filename, 'wb') as handle:
        pickle.dump(binobj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def Plot3D(X,Y,Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    _X, _Y = np.meshgrid(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(_X, _Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    return fig;

def ReadBinaryFile(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

class YSeries():
    def __init__(self, values, line_style='-', points_marker='o', line_color='r', plot_legend_label=None, xvalues=None, legend_marker=None):
        self.values = values
        self.line_style = line_style
        self.points_marker = points_marker
        self.line_color = line_color
        self.plot_legend_label = plot_legend_label
        self.plot_legend = True if plot_legend_label is not None else False
        self.xvalues = xvalues
        if(legend_marker is None):
            self.legend_marker = self.points_marker
        else:
            self.legend_marker = legend_marker
