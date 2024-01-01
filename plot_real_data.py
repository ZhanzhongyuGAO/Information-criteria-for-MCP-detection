import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from functions import detect_change_points


def plt_time_series(time_series, cp_locs, title):
    """
    plot signals and true change point
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_series, color='red', linestyle='dashed', linewidth=0.5, label='Time Series')
    for i in range(0, len(cp_locs)):
        ax.axvline(x=cp_locs[i], color='black', linestyle='solid', linewidth=1.2, alpha=0.8, label='cp_true')
    ax.set_xlabel('Index of Signals')
    ax.set_ylabel('Value of Observations')
    ax.set_title(title)
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='dashed', linewidth=0.5, label='Time Series'),
        Line2D([0], [0], color='black', linestyle='solid', linewidth=1.2, alpha=0.8, label='cp_true'),
    ]
    ax.legend(handles=legend_elements)
    plt.show()


def subplot(ax, time_series, cp_locs, cp_detected, method_name):
    """
    subplot
    """
    ax.plot(time_series, color='red', linestyle='dashed', linewidth=0.5, label='Time Series')
    for i in range(0, len(cp_locs)):
        ax.axvline(x=cp_locs[i], color='black', linestyle='solid', linewidth=1.2, alpha=0.8, label='cp_true')
    for i in range(0, len(cp_detected)):
        ax.axvline(x=cp_detected[i], color='green', linestyle='dashdot', linewidth=0.8, label='cp_detected')
    ax.set_xlabel('Index of time series (' + method_name + ')')
    ax.set_ylabel('Pitch Motor Temperature')
    ax.set_title('number of CPs true/detected: ' + str(len(cp_locs)) + '  / ' + str(len(cp_detected)))


def compare_graph(time_series, cp_locs, df_list):
    """
    plot the compare graph of different information criteria
    """
    fig, axes = plt.subplots(nrows=2, ncols=3,  figsize=(15, 5), sharex=False, sharey=True,)
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axes
    subplot(ax1, time_series, cp_locs, df_list[0], 'aic')
    subplot(ax2, time_series, cp_locs, df_list[1], 'bic')
    subplot(ax3, time_series, cp_locs, df_list[2], 'maic')
    subplot(ax4, time_series, cp_locs, df_list[3], 'mbic1')
    subplot(ax5, time_series, cp_locs, df_list[4], 'mbic2')
    subplot(ax6, time_series, cp_locs, df_list[5], 'mdl')
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='dashed', linewidth=0.5, label='Time Series'),
        Line2D([0], [0], color='black', linestyle='solid', linewidth=1.2, alpha=0.8, label='cp_true'),
        Line2D([0], [0], color='green', linestyle='dashdot', linewidth=0.8, label='cp_detected')
    ]
    ax3.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0., handles=legend_elements)
    fig.tight_layout(h_pad=0.1, w_pad=1)
    plt.show()


if __name__ == "__main__":
    ic_list = ['aic', 'bic', 'maic', 'mbic1', 'mbic2', 'mdl']
    dataset = pd.read_csv('real_data\preprocessed_data.csv')
    time_series = np.array(dataset['0'].dropna())
    time_series1 = np.array(dataset['8'].dropna())
    cp_locs = [357, 493]
    cp_locs1 = [68, 90, 346, 364, 379, 441]
    result_list = []
    for ic in ic_list:
        cp_detected = detect_change_points(time_series1,ic, min_size=50)
        result_list.append(cp_detected)
    plt_time_series(time_series, cp_locs,'Nacelle Temperature')
    compare_graph(time_series, cp_locs, result_list)
    plt_time_series(time_series1, cp_locs1, 'Pitch Motor Temperature')
    compare_graph(time_series1, cp_locs1, result_list)