import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from src.tools.io_tools import read_json

# lines' mark size
set_marker_size = 15
# points' mark size
set_marker_point = 14
# points' mark size
set_font_size = 25
set_lgend_size = 15
set_tick_size = 20
import matplotlib.ticker as ticker

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
base_dir = "../exp_data/"


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


# Function to compute number of parameters for an architecture
def compute_params(architecture):
    layers = [int(layer) for layer in architecture.split('-')]
    params = 0
    for i in range(len(layers) - 1):
        params += layers[i] * layers[i + 1]
    # Add bias terms
    params += sum(layers[1:])
    return params


# Function to convert large number into a string with 'k' for thousands
def func(x, pos):  # formatter function takes tick label and tick position
    if x == 0:
        return f"0"
    else:
        s = f'{x / 1000000}M'
        return s


def draw_parameter_performance():
    # extract train_auc and valid_auc into separate lists
    for dataset, architectures in data_dict.items():
        fig, ax = plt.subplots(figsize=(6.4, 4))
        print(dataset)
        param_sizes = []
        valid_auc = []
        for architecture, epochs in architectures.items():
            for epoch, metrics in epochs.items():
                if str(epoch_sampled[dataset]) == epoch:
                    param_sizes.append(compute_params(architecture))
                    valid_auc.append(metrics["valid_auc"])
                    break

        plt.scatter(param_sizes, valid_auc)
        y_format = ticker.FuncFormatter(func)
        ax.xaxis.set_major_formatter(y_format)
        plt.grid()
        plt.xlabel('Parameter Size')
        plt.ylabel('Validation AUC')
        # plt.legend(loc='upper left', fontsize=set_lgend_size)
        plt.tight_layout()
        export_legend(ori_fig=fig, colnum=5)
        fig.savefig(f"para_{dataset}.jpg", bbox_inches='tight')


dataset_used = "frappe"
# dataset_used = "uci_diabetes"
# dataset_used = "criteo"

epoch_sampled = {"frappe": 19, "uci_diabetes": 35, "criteo": 9}

if dataset_used == "frappe":
    mlp_train_frappe = os.path.join(
        base_dir,
        "tab_data/frappe/all_train_baseline_frappe.json")
    data_dict = read_json(mlp_train_frappe)
elif dataset_used == "uci_diabetes":
    mlp_train_uci_diabetes = os.path.join(
        base_dir,
        "tab_data/uci_diabetes/all_train_baseline_uci_160k_40epoch.json")

    data_dict = read_json(mlp_train_uci_diabetes)
elif dataset_used == "criteo":
    mlp_train_criteo = os.path.join(
        base_dir,
        "tab_data/criteo/all_train_baseline_criteo.json")

    data_dict = read_json(mlp_train_criteo)
else:
    print("err")

draw_parameter_performance()
