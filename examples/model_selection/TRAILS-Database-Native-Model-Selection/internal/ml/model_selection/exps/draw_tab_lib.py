from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import warnings
import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# lines' mark size
set_marker_size = 1
# points' mark size
set_marker_point = 14
# points' mark size
set_font_size = 20
set_lgend_size = 15
set_tick_size = 20

frontinsidebox = 23

# update tick size
matplotlib.rc('xtick', labelsize=set_tick_size)
matplotlib.rc('ytick', labelsize=set_tick_size)

plt.rcParams['axes.labelsize'] = set_tick_size

mark_list = ["o", "*", "<", "^", "s", "d", "D", ">", "h"]
mark_size_list = [set_marker_size, set_marker_size + 1, set_marker_size + 1, set_marker_size,
                  set_marker_size, set_marker_size, set_marker_size, set_marker_size + 1, set_marker_size + 2]
line_shape_list = ['-.', '--', '-', ':']
shade_degree = 0.2


def Add_one_line(x_time_array: list, y_twod_budget: List[List], namespace: str, index, ax):
    # training-based
    x_ = x_time_array
    y_ = y_twod_budget

    if all(isinstance(item, list) for item in x_):
        expx = np.array(x_)
        x_m = np.quantile(expx, .5, axis=0)
    else:
        x_m = x_

    exp = np.array(y_)
    exp = np.where(exp > 10, exp, exp * 100)

    y_h = np.quantile(exp, .75, axis=0)
    y_m = np.quantile(exp, .5, axis=0)
    y_l = np.quantile(exp, .25, axis=0)

    ax.plot(x_m, y_m,
            mark_list[int(index % len(mark_list))] + line_shape_list[int(index % len(line_shape_list))],
            label=namespace,
            markersize=mark_size_list[int(index % len(mark_list))],
            linewidth=3
            )

    ax.fill_between(x_m, y_l, y_h, alpha=shade_degree)
    return x_m


def draw_structure_data_anytime(
        all_lines: List,
        dataset: str, name_img: str, max_value,
        figure_size=(6.4, 4.5),
        annotations=[],
        x_ticks=None, y_ticks=None, unique_labels=None):
    fig, ax = plt.subplots(figsize=figure_size)

    # draw all lines
    time_usage = []
    for i, each_line_info in enumerate(all_lines):
        _x_array = each_line_info[0]
        _y_2d_array = each_line_info[1]
        _name_space = each_line_info[2]
        time_arr = Add_one_line(_x_array, _y_2d_array, _name_space, i, ax)
        time_usage.append(time_arr)

    # print(f"speed-up on {dataset} = {time_usage[0][-1] / time_usage[2][-2]}, "
    #       f"t_train = {time_usage[0][-1]}, t_f = {time_usage[2][-2]}")

    # plt.xscale("log")
    # plt.grid()
    # plt.xlabel(r"Time Budget $T$ (min)", fontsize=set_font_size)
    # plt.ylabel(f"AUC on {dataset.upper()}", fontsize=set_font_size)

    plt.xscale("log")
    ax.grid()
    ax.set_xlabel(r"Response Time Threshold $T_{max}$ (min)", fontsize=set_font_size)
    ax.set_ylabel(f"AUC on {dataset.upper()}", fontsize=set_font_size)
    # ax.set_xscale("log")
    # ax.set_xlim(0.001, 10e4)
    # ax.set_ylim(x1_lim[0], x1_lim[1])

    if y_ticks is not None:
        if y_ticks[0] is not None:
            ax.set_ylim(bottom=y_ticks[0])
        if y_ticks[1] is not None:
            ax.set_ylim(top=y_ticks[1])
        # ax.set_ylim(y_ticks[0], y_ticks[1])
        # ax.set_yticks(y_ticks)
        # ax.set_yticklabels(y_ticks)
    if x_ticks is not None:
        if x_ticks[0] is not None:
            ax.set_xlim(left=x_ticks[0])
        if x_ticks[1] is not None:
            ax.set_xlim(right=x_ticks[1])

    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))

    if max_value > 0:
        plt.axhline(max_value, color='r', linestyle='-', label='Global Best AUC')

    for i in range(len(annotations)):
        ele = annotations[i]
        ax.plot(ele[2], ele[1], mark_list[i], label=ele[0], markersize=set_marker_point)

    # export_legend(fig, filename="any_time_legend", unique_labels=["Training-Based MS", "Training-Free MS", "2Phase-MS", 'Global Best AUC'])
    export_legend(ori_fig=fig, colnum=5, unique_labels=unique_labels)
    plt.tight_layout()

    fig.savefig(f"{name_img}.pdf", bbox_inches='tight')


def export_legend(ori_fig, filename="any_time_legend", colnum=9, unique_labels=None):
    if unique_labels is None:
        unique_labels = []
    fig2 = plt.figure(figsize=(5, 0.3))
    lines_labels = [ax.get_legend_handles_labels() for ax in ori_fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # grab unique labels
    if len(unique_labels) == 0:
        unique_labels = set(labels)
    # assign labels and legends in dict
    legend_dict = dict(zip(labels, lines))
    # query dict based on unique labels
    unique_lines = [legend_dict[x] for x in unique_labels]
    fig2.legend(unique_lines, unique_labels, loc='center',
                ncol=colnum,
                fancybox=True,
                shadow=True, scatterpoints=1, fontsize=set_lgend_size)
    fig2.tight_layout()
    fig2.savefig(f"{filename}.pdf", bbox_inches='tight')


import seaborn as sns
import matplotlib.pyplot as plt


def plot_heatmap(data: List, fontsize: int,
                 x_array_name: str, y_array_name: str,
                 title: str, output_file: str,
                 decimal_places: int,
                 u_ticks, k_ticks,
                 ):
    labelsize = fontsize
    # Convert the data to a NumPy array
    data_array = np.array(data)

    # Custom annotation function
    def custom_annot(val):
        return "{:.{}f}".format(val, decimal_places) if val > 0 else ""

    # Convert the custom annotations to a 2D array
    annot_array = np.vectorize(custom_annot)(data_array)

    # Create a masked array to hide the cells with values less than or equal to 0
    masked_data = np.ma.masked_array(data_array, data_array <= 0)

    # Set the figure size (width, height) in inches
    fig, ax = plt.subplots(figsize=(8, 4))

    # Use the "viridis" colormap
    cmap = "viridis"

    # Create a heatmap
    sns.heatmap(masked_data, annot=annot_array, fmt='', cmap=cmap, mask=masked_data.mask, ax=ax,
                annot_kws={"size": fontsize, "ha": "center", "va": "center"},
                xticklabels=u_ticks, yticklabels=k_ticks)

    # Set axis labels
    ax.set_xlabel(x_array_name, fontsize=fontsize)
    ax.set_ylabel(y_array_name, fontsize=fontsize)

    # Set x/y-axis tick size
    ax.tick_params(axis='both', which='major', labelsize=labelsize)

    # Set the title
    # ax.set_title(title, fontsize=fontsize)

    # Set tight layout
    plt.tight_layout()

    # Save the plot to a PDF file
    plt.savefig(output_file)
