import numpy as np
from ruptures.base import BaseCost


class CustomCost(BaseCost):
    """Custom cost based on maximum likelihood Estimation."""
    model = ''
    min_size = 2

    def fit(self, signal):
        self.signal = signal
        return self


    def error(self, start, end):
        segment = self.signal[start:end+1]
        segment_mean = np.mean(segment)
        segment_var = 1

        # 计算对数似然
        log_likelihood = -0.5 * (end - start + 1) * np.log(2 * np.pi * segment_var) \
                         - 0.5 * np.sum(np.square(segment - segment_mean)) / segment_var

        return -2 * log_likelihood

    def pen_term(self, para_num, pen_name, start, end, tmp_partiton):
        """
        :param para_num: number of model free parameter
        :param pen_name: the name of the information criteria used
        :param start: parameter used in mBIC_1
        :param end: parameter used in mBIC_1
        :param tmp_partiton: parameter used in MDL
        :return: adjustable penalty in pelt
        """
        penalty = None
        if pen_name == 'aic':
            penalty = 2 * para_num * 2
        elif pen_name == 'bic':
            penalty = 2 * para_num * np.log(len(self.signal))
        elif pen_name == 'maic':
            penalty = 2 * para_num + 6
        elif pen_name == 'mbic1':
            penalty = 3 * np.log(len(self.signal)) + np.log((end - start)/len(self.signal))
        elif pen_name == 'mbic2':
            k_current = len(tmp_partiton[-1])
            x = (tmp_partiton[-1][1] - tmp_partiton[-1][0]) / len(self.signal)
            c = 1
            m = (x - 1 / (k_current+1)) ** 2
            penalty = np.log(len(self.signal)) * (2 * para_num + c * m)
        elif pen_name == 'mdl':
            k_previous = None
            k_current = None
            for i in range(0, len(tmp_partiton)):
                if i == 0:
                    k_previous = 0
                else:
                    k_previous = len(tmp_partiton)
                k_current = len(tmp_partiton)
            p1 = np.log(end - start) / 2
            if  k_previous == 0:
                p2 = 0
            else:
                p2 = np.log(k_current/k_previous)
            p3 = np.log(end)
            penalty = 2 * p1 + 2 * p2 + 2 * para_num * p3
        return penalty
