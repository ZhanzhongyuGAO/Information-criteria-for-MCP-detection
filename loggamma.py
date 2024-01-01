import numpy as np
import pandas as pd
import numpy.random
import matplotlib
import functions as fc
import scipy
matplotlib.use('TkAgg')
from scipy.stats import loggamma


def generate_scales(m, delta_scale):
    """
    generate scale list (equivalent mean in loggamma.rvs())
    :param m(int): number of change points
    :param delta_scale(float): change magnitude of scale
    :return: scale list(list)
    """
    digamma = scipy.special.digamma(1.43)
    l1 = 1 - digamma
    l2 = 1 + delta_scale - digamma
    r = np.random.uniform(-0.25, 0.25, m + 1)
    scale_list = []
    for i in range(0, m + 1):
        if i % 2 == 0:
            l = l1 + r[i]
        else:
            l = l2 + r[i]
        scale_list.append(l)
    return scale_list


def loggamma_time_series(segments, m, shape, scales):
    """
    Generate loggamma time series with mean change
    """
    n_segments = m + 1
    time_series = np.empty(sum(segments))
    for i in range(n_segments):
        if i == 0:
            start_index = 0
            end_index = segments[0]
        else:
            start_index = sum(segments[:i])
            end_index = sum(segments[:i+1])
        time_series[start_index:end_index] = loggamma.rvs(shape, loc=scales[i], size=end_index-start_index)
    return time_series


def simulation(ic_list, margin, delta_scale):
    """
    run simulation
    """
    averaged_precisions = pd.DataFrame(index=ic_list)
    averaged_recalls = pd.DataFrame(index=ic_list)
    averaged_ratios = pd.DataFrame(index=ic_list)
    for m in range(1, 21):
        print('Simulation of the number of CPs:', m)
        stds = [1] * (m + 1)
        sample_precision_results = []
        sample_recall_results = []
        sample_ratio_results = []
        for x in range(0, 10):
            shape = 1.43
            scales = generate_scales(m, delta_scale)
            precision_lists = []
            recall_lists = []
            ratio_lists = []
            numpy.random.seed(x)
            segments, cp_locs = fc.generate_observation_sequence(m)
            time_series = loggamma_time_series(segments, m, shape, scales)
            for k in range(0, len(ic_list)):
                cp_detected = fc.detect_change_points(time_series, ic_list[k], para_nums=1)
                precision_score, recall_score = fc.precision_recall(cp_locs, cp_detected, margin)
                precision_lists.append(precision_score)
                recall_lists.append(recall_score)
                if precision_score == 0:
                    ratio_score = 0
                else:
                    ratio_score = recall_score / precision_score
                ratio_lists.append(ratio_score)
            sample_precision_results.append(precision_lists)
            sample_recall_results.append(recall_lists)
            sample_ratio_results.append(ratio_lists)
        averaged_precisions.insert(0, str(m), np.array(sample_precision_results).mean(axis=0))
        averaged_recalls.insert(0, str(m), np.array(sample_recall_results).mean(axis=0))
        averaged_ratios.insert(0, str(m), np.array(sample_ratio_results).mean(axis=0))
    ordered_precisions = averaged_precisions[averaged_precisions.columns[::-1]]
    ordered_recalls = averaged_recalls[averaged_recalls.columns[::-1]]
    ordered_ratios = averaged_ratios[averaged_ratios.columns[::-1]]
    return ordered_precisions, ordered_recalls, ordered_ratios


if __name__ == "__main__":
    ic_list = ['aic', 'bic', 'maic', 'mbic1', 'mbic2', 'mdl']
    margin = 5
    delta_scale = 1.25
    cp_nums = [i for i in range(1, 21)]
    ordered_precisions, ordered_recalls, ordered_ratios = simulation(ic_list, margin, delta_scale)
    fc.draw_score_curve(cp_nums, ic_list, ordered_precisions, 'Precision Rate', 'Independent Log-gamma')
    fc.draw_score_curve(cp_nums, ic_list, ordered_recalls, 'Recall Rate', 'Independent Log-gamma')
    fc.draw_score_curve(cp_nums, ic_list, ordered_ratios, 'Ratio of Change Point Numbers', 'Independent Log-gamma')
