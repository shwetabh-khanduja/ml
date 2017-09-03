import utils as u
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def TestPlotting():
    y1 = u.YSeries(np.arange(10) * 2, line_style='-',
                   points_marker='o', line_color='r', plot_legend_label='x^2')
    y2 = u.YSeries(np.arange(10), line_style='-',
                   points_marker='x', line_color='b', plot_legend_label='x')
    x = np.arange(10)
    fig, ax = u.SaveDataPlotWithLegends([y1, y2], x, r"c:\temp\testfig.png", dispose_fig=False,
                                        x_axis_name="x values", y1_axis_name="y values", title="x square")
    plt.show(fig)

if __name__ == '__main__':
    TestPlotting()
