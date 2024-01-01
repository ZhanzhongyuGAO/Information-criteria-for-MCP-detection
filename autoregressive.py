import numpy as np
import pandas as pd
import numpy.random
import matplotlib
import functions as fc
matplotlib.use('TkAgg')


def generate_means(m, delta_mean, phi):
    """
    generate mean list
    :param m(int): number of change points
    :param delta_mean(float): change magnitude of mean
    :param phi(float): autoregressive coefficient
    :return: mean list(list)
    """
    mu1 = 1 * (1 - phi)
    mu2 = (1 + delta_mean) * (1 - phi)
    r = np.random.uniform(-0.25*(1-phi), 0.25*(1-phi), m + 1)
    mean_list = []
    for i in range(0, m + 1):
        if i % 2 == 0:
            mu = mu1 + r[i]
        else:
            mu = mu2 + r[i]
        mean_list.append(mu)
    return mean_list


def generate_ar_series(segments, m, means, phi):
    """
    Generate ar(1) time series with gaussian innovation(mean change)
    """
    std = np.sqrt(1 - phi**2)
    n_segments = m + 1
    time_series = np.empty(sum(segments))
    error = np.random.normal(0, std, len(time_series))
    for i in range(n_segments):
        if i == 0:
            start_index = 0
            end_index = segments[0]
        else:
            start_index = sum(segments[:i])
            end_index = sum(segments[:i + 1])
        for j in range(start_index,end_index):
            if j == 0:
                time_series[j] = means[i] + error[j]
            else:
                time_series[j] = means[i] + error[j] + phi * time_series[j-1]
    return time_series


def simulation(ic_list, margin, delta_mean, phi):
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
        for x in range(0, 1):
            means = generate_means(m, delta_mean, phi)
            precision_lists = []
            recall_lists = []
            ratio_lists = []
            numpy.random.seed(x)
            segments, cp_locs = fc.generate_observation_sequence(m)
            time_series = generate_ar_series(segments, m, means[:m+1], phi)
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
    phi = 0.5
    delta_mean = 1.25
    cp_nums = [i for i in range(1, 21)]
    ordered_precisions, ordered_recalls, ordered_ratios = simulation(ic_list, margin, delta_mean, phi)
    fc.draw_score_curve(cp_nums, ic_list, ordered_precisions, 'Precision Rate', r'Normal AR(1) with $\phi=0.5$')
    fc.draw_score_curve(cp_nums, ic_list, ordered_recalls, 'Recall Rate', r'Normal AR(1) with $\phi=0.5$')
    fc.draw_score_curve(cp_nums, ic_list, ordered_ratios, 'Ratio of Change Point Numbers', r'Normal AR(1) with $\phi=0.5$')
