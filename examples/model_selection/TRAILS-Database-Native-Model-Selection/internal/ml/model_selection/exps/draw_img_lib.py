import os

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import palettable
from matplotlib.ticker import MaxNLocator
import numpy
from src.common.constant import Config
import matplotlib

# lines' mark size
set_marker_size = 15
# points' mark size
set_marker_point = 14
# points' mark size
set_font_size = 40
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


# this is for draw figure3 only
def get_plot_compare_with_base_line_cfg(search_space, dataset, if_with_phase1=False):
    if search_space == Config.NB201:
        run_range_ = range(0, 100, 1)
        if if_with_phase1:
            draw_graph = draw_anytime_result_with_p1
        else:
            draw_graph = draw_anytime_result
        # min, this is for plot only
        if dataset == Config.c10:
            # C10 array
            budget_array = [0.017, 0.083] + list(range(1, 350, 4))
            sub_graph_y1 = [91, 94.5]
            sub_graph_y2 = [53.5, 55]
            sub_graph_split = 60
        elif dataset == Config.c100:
            # C10 array
            budget_array = [0.017, 0.083] + list(range(1, 350, 4))

            sub_graph_y1 = [64, 73.5]
            sub_graph_y2 = [15, 16]
            sub_graph_split = 20
        else:
            # ImgNet X array
            budget_array = [0.017, 0.083] + list(range(1, 350, 4))
            sub_graph_y1 = [33, 48]
            sub_graph_y2 = [15.5, 17]
            sub_graph_split = 34
    else:
        # this is NB101 + C10, because only 101 has 20 run. others have 100 run.
        run_range_ = range(0, 20, 1)
        if if_with_phase1:
            draw_graph = draw_anytime_result_one_graph_with_p1
            # budget_array = list(range(1, 16, 1))
            budget_array = numpy.arange(0.02, 15, 0.02).tolist()
        else:
            draw_graph = draw_anytime_result_one_graph
            budget_array = [0.017, 0.083] + list(range(1, 2000, 8))

        if dataset == Config.c10:
            # C10 array
            # budget_array = list(range(0, 2000, 1))
            sub_graph_y1 = [90, 94.5]
            sub_graph_y2 = [52, 55]
            sub_graph_split = 60
        else:
            raise Exception

    return run_range_, budget_array, sub_graph_y1, sub_graph_y2, sub_graph_split, draw_graph


def draw_anytime_result(result_dir, y_acc_list_arr, x_T_list,
                        x_acc_train, y_acc_train_l, y_acc_train_m, y_acc_train_h,
                        annotations, lv,
                        name_img, dataset,
                        x1_lim=[], x2_lim=[],
                        ):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=100, gridspec_kw={'height_ratios': [4, 1]})
    exp = np.array(y_acc_list_arr)
    sys_acc_h = np.quantile(exp, .75, axis=0)
    sys_acc_m = np.quantile(exp, .5, axis=0)
    sys_acc_l = np.quantile(exp, .25, axis=0)

    # plot simulate result of system
    ax1.fill_between(x_T_list, sys_acc_l, sys_acc_h, alpha=0.1)
    ax1.plot(x_T_list, sys_acc_m, mark_list[-1], label="TRAILS")
    ax2.fill_between(x_T_list, sys_acc_l, sys_acc_h, alpha=0.1)

    # plot simulate result of train-based line
    ax1.fill_between(x_acc_train, y_acc_train_l, y_acc_train_h, alpha=0.3)
    ax1.plot(x_acc_train, y_acc_train_m, mark_list[-2], label="Training-based MS")
    ax2.fill_between(x_acc_train, y_acc_train_l, y_acc_train_h, alpha=0.3)

    for i in range(len(annotations)):
        ele = annotations[i]
        if ele[1] < lv:
            # convert to mins
            ax2.plot(ele[2] / 60, ele[1], mark_list[i], label=ele[0], fontsize=set_marker_size)
        else:
            ax1.plot(ele[2] / 60, ele[1], mark_list[i], label=ele[0], fontsize=set_marker_size)
        # ax2.scatter(ele[2]/60, ele[1]* 0.01, s=100, color="red")
        # ax2.annotate(ele[0], (ele[2]/60, ele[1] * 0.01))

    if len(x1_lim) > 0 and len(x2_lim) > 0:
        ax1.set_ylim(x1_lim[0], x1_lim[1])  # 子图1设置y轴范围，只显示部分图
        ax2.set_ylim(x2_lim[0], x2_lim[1])  # 子图2设置y轴范围，只显示部分图

    ax1.spines['bottom'].set_visible(False)  # 关闭子图1中底部脊
    ax2.spines['top'].set_visible(False)  ##关闭子图2中顶部脊
    ax2.set_xticks(range(0, 31, 1))

    d = .85  # 设置倾斜度
    # 绘制断裂处的标记
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=set_marker_size,
                  linestyle='none', color='r', mec='r', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    plt.tight_layout()
    plt.xscale("symlog")
    ax1.grid()
    ax2.grid()
    plt.xlabel("Time Budget given by user (min)", fontsize=set_font_size)
    ax1.set_ylabel(f"Test accuracy on {dataset}", fontsize=set_font_size)
    ax1.legend(ncol=1, fontsize=set_lgend_size)
    ax2.legend(fontsize=set_lgend_size)
    # plt.show()
    plt.savefig(f"{result_dir}/any_time_{name_img}.pdf", bbox_inches='tight')


def draw_anytime_result_one_graph(y_acc_list_arr, x_T_list,
                                  x_acc_train, y_acc_train_l, y_acc_train_m, y_acc_train_h,
                                  annotations, lv,
                                  name_img, dataset,
                                  x1_lim=[], x2_lim=[],
                                  ):
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=100, gridspec_kw={'height_ratios': [5, 1]})
    exp = np.array(y_acc_list_arr) * 100
    sys_acc_h = np.quantile(exp, .75, axis=0)
    sys_acc_m = np.quantile(exp, .5, axis=0)
    sys_acc_l = np.quantile(exp, .25, axis=0)

    # exp_time = np.array(real_time_used_arr)
    # time_mean = np.quantile(exp_time, .5, axis=0)
    time_mean = x_T_list

    # plot simulate result of system
    plt.fill_between(time_mean, sys_acc_l, sys_acc_h, alpha=0.1)
    plt.plot(time_mean, sys_acc_m, "o-", label="TRAILS")
    # plt.plot(time_mean, sys_acc_m, label="TRAILS")

    # plot simulate result of train-based line
    plt.fill_between(x_acc_train, y_acc_train_l, y_acc_train_h, alpha=0.3)
    plt.plot(x_acc_train, y_acc_train_m, "o-", label="Training-based MS")
    # plt.plot(x_acc_train, y_acc_train_m,  label="Training-based MS")

    if len(x1_lim) > 0:
        plt.ylim(x1_lim[0], x1_lim[1])  # 子图1设置y轴范围，只显示部分图

    d = .85  # 设置倾斜度
    # 绘制断裂处的标记
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=set_marker_size,
                  linestyle='none', color='r', mec='r', mew=1, clip_on=False)
    # plt.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    # plt.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    plt.tight_layout()
    # plt.xscale("symlog")
    plt.grid()
    plt.xlabel("Time Budget given by user (min)", fontsize=set_font_size)
    plt.ylabel(f"Test accuracy on {dataset}", fontsize=set_font_size)
    plt.legend(ncol=1, fontsize=set_lgend_size)
    plt.show()
    # plt.savefig(f"amy_time_{name_img}.pdf", bbox_inches='tight')


# those two function will plot phase 1 and phase 2
def draw_anytime_result_with_p1(result_dir, y_acc_list_arr, x_T_list, y_acc_list_arr_p1, x_T_list_p1,
                                x_acc_train, y_acc_train_l, y_acc_train_m, y_acc_train_h,
                                annotations, lv,
                                name_img, dataset, max_value,
                                x1_lim=[], x2_lim=[],
                                ):
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        sharex=True,
        dpi=100,
        gridspec_kw={'height_ratios': [6, 1]})

    shade_degree = 0.2

    # plot simulate result of train-based line
    ax1.plot(x_acc_train, y_acc_train_m, mark_list[-3] + line_shape_list[0], label="Training-Based MS",
             markersize=mark_size_list[-3])
    ax1.fill_between(x_acc_train, y_acc_train_l, y_acc_train_h, alpha=shade_degree)
    ax2.fill_between(x_acc_train, y_acc_train_l, y_acc_train_h, alpha=shade_degree)

    # plot simulate result of system
    exp = np.array(y_acc_list_arr_p1)
    sys_acc_p1_h = np.quantile(exp, .75, axis=0)
    sys_acc_p1_m = np.quantile(exp, .5, axis=0)
    sys_acc_p1_l = np.quantile(exp, .25, axis=0)
    ax1.plot(x_T_list_p1, sys_acc_p1_m, mark_list[-2] + line_shape_list[1], label="Training-Free MS",
             markersize=mark_size_list[-2])
    ax1.fill_between(x_T_list_p1, sys_acc_p1_l, sys_acc_p1_h, alpha=shade_degree)
    ax2.fill_between(x_T_list_p1, sys_acc_p1_l, sys_acc_p1_h, alpha=shade_degree)

    # plot simulate result of system
    exp = np.array(y_acc_list_arr)
    sys_acc_h = np.quantile(exp, .75, axis=0)
    sys_acc_m = np.quantile(exp, .5, axis=0)
    sys_acc_l = np.quantile(exp, .25, axis=0)
    ax1.plot(x_T_list, sys_acc_m, mark_list[-1] + line_shape_list[2], label="2Phase-MS", markersize=mark_size_list[-1])
    ax1.fill_between(x_T_list, sys_acc_l, sys_acc_h, alpha=shade_degree)
    ax2.fill_between(x_T_list, sys_acc_l, sys_acc_h, alpha=shade_degree)

    print(f"speed-up on {dataset} = {x_acc_train[-1] / x_T_list[-2]}, "
          f"t_train = {x_acc_train[-1]}, t_f = {x_T_list[-2]}")

    for i in range(len(annotations)):
        ele = annotations[i]
        if ele[1] < lv:
            # convert to mins
            ax2.plot(ele[2] / 60, ele[1], mark_list[i], label=ele[0], markersize=set_marker_point)
        else:
            ax1.plot(ele[2] / 60, ele[1], mark_list[i], label=ele[0], markersize=set_marker_point)
        # ax2.scatter(ele[2]/60, ele[1]* 0.01, s=100, color="red")
        # ax2.annotate(ele[0], (ele[2]/60, ele[1] * 0.01))

    if len(x1_lim) > 0 and len(x2_lim) > 0:
        ax1.set_ylim(x1_lim[0], x1_lim[1])  # 子图1设置y轴范围，只显示部分图
        ax2.set_ylim(x2_lim[0], x2_lim[1])  # 子图2设置y轴范围，只显示部分图

    ax1.spines['bottom'].set_visible(False)  # 关闭子图1中底部脊
    ax2.spines['top'].set_visible(False)  ##关闭子图2中顶部脊
    ax2.set_xticks(range(0, 31, 1))

    d = .85  # 设置倾斜度
    # 绘制断裂处的标记
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=set_marker_size,
                  linestyle='none', color='r', mec='r', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    plt.xscale("log")
    ax1.grid()
    ax2.grid()
    plt.xlabel(r"Response Time Threshold $T_{max}$ (min)", fontsize=set_font_size)
    ax1.set_ylabel(f"Test Acc on {'In-16'}", fontsize=set_font_size)
    # ax1.legend(ncol=1, fontsize=set_lgend_size)
    # ax2.legend(fontsize=set_lgend_size)

    ax1.xaxis.label.set_size(set_tick_size)
    ax1.yaxis.label.set_size(set_tick_size)
    # ax1.set_xticks([])

    ax2.xaxis.label.set_size(set_tick_size)
    ax2.yaxis.label.set_size(set_tick_size)

    ax1.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

    ax1.axhline(max_value, color='r', linestyle='-', label='Global Best Accuracy')

    tick_values = [0.01, 0.1, 1, 10, 100, 1000]
    ax2.set_xticks(tick_values)
    ax2.set_xticklabels([f'$10^{{{int(np.log10(val))}}}$' for val in tick_values])

    # this is for unique hash
    export_legend(
        fig,
        colnum=3,
        unique_labels=['TE-NAS (Training-Free)', 'ENAS (Weight sharing)',
                       'KNAS (Training-Free)', 'DARTS-V1 (Weight sharing)', 'DARTS-V2 (Weight sharing)',
                       'Training-Based MS', 'Training-Free MS', '2Phase-MS', 'Global Best Accuracy'])
    plt.tight_layout()
    fig.savefig(f"{result_dir}/any_time_{name_img}_p1_from_0.1_sec.pdf", bbox_inches='tight')


def export_legend(ori_fig, filename="any_time_legend", colnum=9, unique_labels=[]):
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


def draw_anytime_result_one_graph_with_p1(y_acc_list_arr, x_T_list, y_acc_list_arr_p1, x_T_list_p1,
                                          x_acc_train, y_acc_train_l, y_acc_train_m, y_acc_train_h,
                                          annotations, lv,
                                          name_img, dataset,
                                          x1_lim=[], x2_lim=[],
                                          ):
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=100, gridspec_kw={'height_ratios': [5, 1]})

    # plot simulate result of system
    exp = np.array(y_acc_list_arr_p1) * 100
    sys_acc_p1_h = np.quantile(exp, .75, axis=0)
    sys_acc_p1_m = np.quantile(exp, .5, axis=0)
    sys_acc_p1_l = np.quantile(exp, .25, axis=0)

    plt.fill_between(x_T_list_p1, sys_acc_p1_l, sys_acc_p1_h, alpha=0.1)
    plt.plot(x_T_list_p1, sys_acc_p1_m, "o-", label="TRAILS-P1")
    # plt.fill_between(x_T_list_p1, sys_acc_p1_l, sys_acc_p1_h, alpha=0.1)

    exp = np.array(y_acc_list_arr) * 100
    sys_acc_h = np.quantile(exp, .75, axis=0)
    sys_acc_m = np.quantile(exp, .5, axis=0)
    sys_acc_l = np.quantile(exp, .25, axis=0)

    # exp_time = np.array(real_time_used_arr)
    # time_mean = np.quantile(exp_time, .5, axis=0)
    time_mean = x_T_list

    # plot simulate result of system
    plt.fill_between(time_mean, sys_acc_l, sys_acc_h, alpha=0.1)
    plt.plot(time_mean, sys_acc_m, "o-", label="TRAILS")
    # plt.plot(time_mean, sys_acc_m, label="TRAILS")

    # plot simulate result of train-based line
    plt.fill_between(x_acc_train, y_acc_train_l, y_acc_train_h, alpha=0.3)
    plt.plot(x_acc_train, y_acc_train_m, "o-", label="Training-based MS")
    # plt.plot(x_acc_train, y_acc_train_m,  label="Training-based MS")

    if len(x1_lim) > 0:
        plt.ylim(x1_lim[0], x1_lim[1])  # 子图1设置y轴范围，只显示部分图

    d = .85  # 设置倾斜度
    # 绘制断裂处的标记
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=set_marker_size,
                  linestyle='none', color='r', mec='r', mew=1, clip_on=False)
    # plt.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    # plt.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    plt.tight_layout()
    plt.xscale("symlog")
    plt.grid()
    plt.xlabel("Time Budget given by user (min)", fontsize=set_font_size)
    plt.ylabel(f"Test accuracy on {dataset}", fontsize=set_font_size)
    plt.legend(ncol=1, fontsize=set_lgend_size)
    # plt.show()
    plt.savefig(f"amy_time_{name_img}.pdf", bbox_inches='tight')


# for K, U N trade-off
def draw_grid_graph_with_budget(
        acc, bt, b1, b2,
        img_name: str, y_array: list, x_array: list):
    """
    :param acc: Two array list
    :param bt:  Two array list
    :param img_name: img name string
    :return:
    """

    acc_new = np.array(acc)
    acc = acc_new.tolist()

    mask = np.array(acc)
    mask[mask > 0] = 0
    mask[mask < 0] = 1

    bt = np.round(np.array(bt), 2).tolist()
    mask2 = np.array(bt)
    mask2[mask2 > 0] = 0
    mask2[mask2 < 0] = 1

    mask3 = np.array(b1)
    mask3[mask3 > 0] = 0
    mask3[mask3 < 0] = 1

    mask4 = np.array(b2)
    mask4[mask4 > 0] = 0
    mask4[mask4 < 0] = 1

    fig, ax = plt.subplots(2, 2, figsize=(15, 14))

    linewidths = 0.5
    sns.set(font_scale=3)
    sns.heatmap(
        data=acc,
        vmax=99,
        vmin=93,
        cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
        annot=True,
        fmt=".2f",
        annot_kws={'size': frontinsidebox, 'weight': 'normal', 'color': 'w', 'va': 'bottom'},
        mask=mask,
        square=True, linewidths=linewidths,  # 每个方格外框显示，外框宽度设置
        cbar_kws={"shrink": .5},
        ax=ax[0, 0]
    )

    sns.heatmap(
        data=bt,
        # vmax=,
        vmin=-9,
        cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
        annot=True,
        fmt=".2f",
        annot_kws={'size': frontinsidebox, 'weight': 'normal', 'color': 'w', 'va': 'top'},
        mask=mask2,
        square=True, linewidths=linewidths,  # 每个方格外框显示，外框宽度设置
        cbar_kws={"shrink": .5},
        ax=ax[0, 1]
    )

    sns.heatmap(
        data=b1,
        vmax=17000,
        vmin=15000,
        cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
        annot=True,
        fmt=".0f",
        annot_kws={'size': frontinsidebox, 'weight': 'normal', 'color': 'w', 'va': 'top'},
        mask=mask4,
        square=True, linewidths=linewidths,  # 每个方格外框显示，外框宽度设置
        cbar_kws={"shrink": .5},
        ax=ax[1, 0]
    )

    sns.heatmap(
        data=b2,
        # vmax=,
        # vmin=-9,
        cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
        annot=True,
        fmt=".0f",
        annot_kws={'size': frontinsidebox, 'weight': 'normal', 'color': 'w', 'va': 'top'},
        mask=mask4,
        square=True, linewidths=linewidths,  # 每个方格外框显示，外框宽度设置
        cbar_kws={"shrink": .5},
        ax=ax[1, 1]
    )

    plt.tight_layout()
    plt.xlabel("U (epoch)", fontsize=set_font_size)
    plt.ylabel("K (# models)", fontsize=set_font_size)

    for i in [0, 1]:
        for j in [0, 1]:
            ax[i, j].set_xticklabels(x_array, fontsize=set_font_size)
            ax[i, j].set_yticklabels(y_array, fontsize=set_font_size)
            ax[i, j].set_xlabel("U (# epoch)", fontsize=set_font_size)
            ax[i, j].set_ylabel("K (# models)", fontsize=set_font_size)

    ax[0, 0].set_title('Test Accuracy (%)', fontsize=set_font_size)
    ax[0, 1].set_title(r'Time Budget $T$ (min)', fontsize=set_font_size)
    ax[1, 0].set_title(r'$N$', fontsize=set_font_size)
    ax[1, 1].set_title(r"$K \cdot U \cdot \log_{\eta}K$", fontsize=set_font_size)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.001, hspace=0.3)

    # plt.show()
    base_dr = os.getcwd()
    path_gra = os.path.join(base_dr, f"{img_name}.pdf")
    fig.savefig(path_gra, bbox_inches='tight')


def draw_grid_graph_with_budget_only_Acc_and_T(
        acc, bt, b1, b2,
        img_name: str, y_array: list, x_array: list):
    """
    :param acc: Two array list
    :param bt:  Two array list
    :param img_name: img name string
    :return:
    """

    acc_new = np.array(acc)
    acc = acc_new.tolist()

    mask = np.array(acc)
    mask[mask > 0] = 0
    mask[mask < 0] = 1

    bt = np.round(np.array(bt), 2).tolist()
    mask2 = np.array(bt)
    mask2[mask2 > 0] = 0
    mask2[mask2 < 0] = 1

    mask3 = np.array(b1)
    mask3[mask3 > 0] = 0
    mask3[mask3 < 0] = 1

    mask4 = np.array(b2)
    mask4[mask4 > 0] = 0
    mask4[mask4 < 0] = 1

    fig, ax = plt.subplots(1, 2, figsize=(15, 14))

    linewidths = 0.5
    sns.set(font_scale=2)
    sns.heatmap(
        data=acc,
        vmax=99,
        vmin=93,
        cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
        annot=True,
        fmt=".2f",
        annot_kws={'size': frontinsidebox, 'weight': 'normal', 'color': 'w', 'va': 'bottom'},
        mask=mask,
        square=True,
        linewidths=linewidths,  # 每个方格外框显示，外框宽度设置
        cbar_kws={"shrink": .4},
        ax=ax[0]
    )

    sns.heatmap(
        data=bt,
        vmax=600,
        # vmin=-9,
        cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
        annot=True,
        fmt=".2f",
        annot_kws={'size': frontinsidebox, 'weight': 'normal', 'color': 'w', 'va': 'top'},
        mask=mask2,
        square=True,
        linewidths=linewidths,  # 每个方格外框显示，外框宽度设置
        cbar_kws={"shrink": .4},
        ax=ax[1]
    )

    plt.tight_layout()
    plt.xlabel("U (epoch)", fontsize=set_font_size)
    plt.ylabel("K (# models)", fontsize=set_font_size)

    for j in [0, 1]:
        ax[j].set_xticklabels(x_array, fontsize=set_font_size)
        ax[j].set_yticklabels(y_array, fontsize=set_font_size)
        ax[j].set_xlabel("U (# epoch)", fontsize=set_font_size)
        ax[j].set_ylabel("K (# models)", fontsize=set_font_size)

    ax[0].set_title('Test Accuracy (%)', fontsize=set_font_size)
    ax[1].set_title(r'Time Budget $T$ (min)', fontsize=set_font_size)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    # plt.show()
    base_dr = os.getcwd()
    path_gra = os.path.join(base_dr, f"{img_name}.pdf")
    fig.savefig(path_gra, bbox_inches='tight')


def draw_grid_graph_with_budget_only_Acc(
        acc, bt, b1, b2,
        img_name: str, y_array: list, x_array: list):
    """
    :param acc: Two array list
    :param bt:  Two array list
    :param img_name: img name string
    :return:
    """

    acc_new = np.array(acc)
    acc = acc_new.tolist()

    mask = np.array(acc)
    mask[mask > 0] = 0
    mask[mask < 0] = 1

    fig = plt.figure(figsize=(7, 14))

    linewidths = 0.5
    sns.set(font_scale=2)
    sns.heatmap(
        data=acc,
        vmax=99,
        vmin=93,
        cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
        annot=True,
        fmt=".2f",
        annot_kws={'size': frontinsidebox, 'weight': 'normal', 'color': 'w', 'va': 'bottom'},
        mask=mask,
        square=True,
        linewidths=linewidths,  # 每个方格外框显示，外框宽度设置
        cbar_kws={"shrink": .4},
        ax=fig
    )

    plt.tight_layout()
    plt.xlabel("U (epoch)", fontsize=set_font_size)
    plt.ylabel("K (# models)", fontsize=set_font_size)

    plt.xticks(x_array, fontsize=set_font_size)
    plt.yticks(y_array, fontsize=set_font_size)

    plt.title('Test Accuracy (%)', fontsize=set_font_size)
    plt.tight_layout()
    # fig.subplots_adjust(wspace=0.3, hspace=0.3)
    # plt.show()
    base_dr = os.getcwd()
    path_gra = os.path.join(base_dr, f"{img_name}.pdf")
    fig.savefig(path_gra, bbox_inches='tight')


def draw_grid_graph_with_budget_only_T(
        acc, bt, b1, b2,
        img_name: str, y_array: list, x_array: list):
    """
    :param acc: Two array list
    :param bt:  Two array list
    :param img_name: img name string
    :return:
    """

    acc_new = np.array(acc)
    acc = acc_new.tolist()

    mask = np.array(acc)
    mask[mask > 0] = 0
    mask[mask < 0] = 1

    bt = np.round(np.array(bt), 2).tolist()
    mask2 = np.array(bt)
    mask2[mask2 > 0] = 0
    mask2[mask2 < 0] = 1

    mask3 = np.array(b1)
    mask3[mask3 > 0] = 0
    mask3[mask3 < 0] = 1

    mask4 = np.array(b2)
    mask4[mask4 > 0] = 0
    mask4[mask4 < 0] = 1

    fig, ax = plt.subplots(1, 2, figsize=(15, 14))

    linewidths = 0.5
    sns.set(font_scale=2)
    sns.heatmap(
        data=acc,
        vmax=99,
        vmin=93,
        cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
        annot=True,
        fmt=".2f",
        annot_kws={'size': frontinsidebox, 'weight': 'normal', 'color': 'w', 'va': 'bottom'},
        mask=mask,
        square=True,
        linewidths=linewidths,  # 每个方格外框显示，外框宽度设置
        cbar_kws={"shrink": .4},
        ax=ax[0]
    )

    sns.heatmap(
        data=bt,
        vmax=600,
        # vmin=-9,
        cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
        annot=True,
        fmt=".2f",
        annot_kws={'size': frontinsidebox, 'weight': 'normal', 'color': 'w', 'va': 'top'},
        mask=mask2,
        square=True,
        linewidths=linewidths,  # 每个方格外框显示，外框宽度设置
        cbar_kws={"shrink": .4},
        ax=ax[1]
    )

    plt.tight_layout()
    plt.xlabel("U (epoch)", fontsize=set_font_size)
    plt.ylabel("K (# models)", fontsize=set_font_size)

    for j in [0, 1]:
        ax[j].set_xticklabels(x_array, fontsize=set_font_size)
        ax[j].set_yticklabels(y_array, fontsize=set_font_size)
        ax[j].set_xlabel("U (# epoch)", fontsize=set_font_size)
        ax[j].set_ylabel("K (# models)", fontsize=set_font_size)

    ax[0].set_title('Test Accuracy (%)', fontsize=set_font_size)
    ax[1].set_title(r'Time Budget $T$ (min)', fontsize=set_font_size)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    # plt.show()
    base_dr = os.getcwd()
    path_gra = os.path.join(base_dr, f"{img_name}.pdf")
    fig.savefig(path_gra, bbox_inches='tight')
