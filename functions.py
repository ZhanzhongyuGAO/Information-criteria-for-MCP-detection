"""functions used in the simulation"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pelt import Pelt
from matplotlib.ticker import MultipleLocator
from custom_cost import CustomCost
from itertools import product

def generate_dirichlet_samples(size, alpha=[1]*10):
    """
    Generate dirichlet p_values
    :param size(int): number of dirichlet samples
    :param alpha(float): flat parameter used in dirichlet distribution
    :return:
    """
    samples = np.random.dirichlet(alpha, size=size)
    return samples

def generate_observation_sequence(m, l=50):
    """
    :param m(int): number of change point
    :param l(int): lower bound of each segment
    :return: segments(list): length of each segment
    :return: cp_locs(list): locations of change points
    """
    n = l * (m+1)
    p_lists = generate_dirichlet_samples(m, alpha=[1]*(m+1))[0]
    multinomial_samples = np.random.multinomial(n, p_lists)
    segments = [l + sample for sample in multinomial_samples]
    cp_locs = []
    for j in range(1, len(segments)):
        tao_j = sum(segments[0:j])
        cp_locs.append(tao_j)
    return segments, cp_locs

def detect_change_points(time_series, ic, min_size=2, para_nums=1):
    """
    return the detected change points locations
    :param time_series(np.array): generated time series
    :param ic(str): name of used information criteria
    :param min_size(int): mininum distance between change points(required by pelt, d=2 by default)
    :param para_nums(int): number of model changing parameters
    :return: cp_locs(list): the location of change points
    """
    custom_cost = CustomCost()
    algo = Pelt(custom_cost=custom_cost, min_size=min_size, jump=1, para_nums=para_nums).fit(time_series)
    cp_locs = algo.predict(pen=ic)
    return cp_locs[:-1]

def precision_recall(cp_locs, cp_detected, margin):
    """
    Calculate precision and recall rate
    :param cp_locs(list): the locations of true change points
    :param cp_detected(list): the locations of detected change points
    :param margin(int): degree of detection tolerance
    :return: precision(float): precision rate
    :return: recall(float): recall rate
    """
    used = set()
    true_pos = set(
        true_b
        for true_b, my_b in product(cp_locs, cp_detected)
        if my_b - margin <= true_b <= my_b + margin
        and not (my_b in used or used.add(my_b))
    )
    tp_ = len(true_pos)
    if len(cp_detected) == 0 or len(cp_locs) == 0:
        precision = 0
    else:
        precision = tp_ / len(cp_detected)
    recall = tp_ / len(cp_locs)
    return precision, recall

def draw_score_curve(cp_nums, ic_list, score_df, score_name, distribution):
    """
    draw graph
    :param cp_nums:
    :param ic_list:
    :param score_df:
    :param score_name:
    :param distribution:
    :return:
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    color_list = ['crimson', 'lightcoral', 'orange', 'palegreen', 'deepskyblue', 'violet']
    ic_official_name = ['AIC', 'BIC', 'mAIC', 'mBIC$_1$', 'mBIC$_2$', 'MDL']
    marker_list = ["o", "x", "v", "d", "+", "*"]
    for i in range(0, len(ic_list)):
        ax.plot(cp_nums, score_df.loc[ic_list[i]], color=color_list[i], linestyle='dashdot', linewidth=1,
                label=ic_official_name[i], marker=marker_list[i], alpha=0.7)

    y_major_locator = MultipleLocator(0.1)
    x_major_locator = MultipleLocator(1)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.set_xlabel('Index of Time Series')
    ax.set_ylabel(score_name)
    ax.set_title(distribution)
    ax.legend()
    plt.show()
