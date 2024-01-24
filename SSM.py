from scipy.fft import fft
from torch.utils.data import DataLoader, Dataset
import os
import torch
import cv2
import numpy as np
from scipy.signal import hilbert, butter, filtfilt
import matplotlib.pyplot as plt
import math
from scipy import signal
from scipy.signal import find_peaks, detrend
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

# def cosine_similarity(list1, list2):
#     dot_product = sum([list1[i]*list2[i] for i in range(len(list1))])
#     norm1 = math.sqrt(sum([x**2 for x in list1]))
#     norm2 = math.sqrt(sum([x**2 for x in list2]))
#     similarity = dot_product / (norm1 * norm2)
#     return similarity

def my_self_similarity_calc(ippg, Lw):
    L0 = len(ippg) - Lw + 1
    # print(L0)
    sl = []
    for i in range(0,L0):
        sl.append(ippg[i:i+Lw-1])
    # print(len(sl))
    # result_list = []
    result = cosine_similarity(sl, sl)
    # for i in range(0,L0-1):
    #     tmp = []
    #     for j in range(0,L0-1):
    #         tmp.append(cosine_similarity(sl[i], sl[j]))
    #         # tmp.append(np.cos(sl[i] - sl[j]))
    #     tmp_list = torch.FloatTensor(np.array(tmp)).unsqueeze(-1)
    #     result_list.append(tmp_list)
    #     # print(tmp_list.shape)
    # result = torch.cat(result_list, dim=-1)
    return result

def self_similarity_calc(ippg):
    ippg_phase0 = myhilbert(ippg)
    ippg_phase = amass_hilbort(ippg_phase0)[1:]
    result_list = []
    for i in range(len(ippg_phase)):
        tmp_list = []
        for j in range(len(ippg_phase)):
            similarity = np.cos(ippg_phase[i] - ippg_phase[j])
            tmp_list.append(similarity)
        tmp_list = torch.FloatTensor(tmp_list).unsqueeze(-1)
        result_list.append(tmp_list)
    result = torch.cat(result_list, dim=-1)
    return result

def amass_hilbort(ippg):
    peak_record = [(0, 0)]
    current_sum = 0
    for i in range(1, len(ippg)):
        if (ippg[i] - ippg[i - 1]) < 0:
            current_sum += ippg[i - 1] - ippg[i]  # 加入差值
            peak_record.append((i, current_sum))

    sum_list = [peak_record[i][1] for i in range(len(peak_record))]
    peak_record.append((len(ippg), None))
    record_list = [(peak_record[i][0], peak_record[i + 1][0]) for i in range(len(peak_record) - 1)]
    result =[]
    for i in range(len(ippg)):
        for j in range(len(record_list)):
            if record_list[j][0] <= i < record_list[j][1]:
                ans = ippg[i] + sum_list[j]
                while ans > 2 * np.pi:
                    ans -= 2 * np.pi
                result.append(ans)

    return result

def myhilbert(ippg_test):
    """

    :param ippg_test:
    :return: Returns the Hilbert transformed phase information of the input array
    """
    ippg_hilbert = hilbert(ippg_test)
    N = len(ippg_test)
    ippg_hilbert_phase = np.zeros(N)
    ippg_hilbert_phase_shift = np.zeros(N)
    for i in range(N):
        if ippg_hilbert[i].real == 0:
            if (ippg_hilbert[i].imag > 0):
                ippg_hilbert_phase[i] = 1.57
            elif (ippg_hilbert[i].imag < 0):
                ippg_hilbert_phase[i] = -1.57
            else:
                ippg_hilbert_phase[i] = 0
        else:
            ippg_hilbert_phase[i] = math.atan(ippg_hilbert[i].imag / ippg_hilbert[i].real)
    k = 1
    for i in range(N):
        if i != 0 and ippg_hilbert_phase[i] - ippg_hilbert_phase[i - 1] < -1.5:
            k = -k
        ippg_hilbert_phase_shift[i] = k * ippg_hilbert_phase[i]
    return ippg_hilbert_phase


def butter_bandpass(sig, lowcut, highcut, fs, order=2):
    # butterworth bandpass filter

    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    y = filtfilt(b, a, sig)
    return y

def hr_fft(sig, fs=30, harmonics_removal=True):
    # get heart rate by FFT
    # return both heart rate and PSD

    sig = sig.reshape(-1)
    sig = sig * signal.windows.hann(sig.shape[0])
    sig_f = np.abs(fft(sig))
    low_idx = np.round(0.6 / fs * sig.shape[0]).astype('int')
    high_idx = np.round(4 / fs * sig.shape[0]).astype('int')
    sig_f_original = sig_f.copy()

    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0

    peak_idx, _ = signal.find_peaks(sig_f)
    sort_idx = np.argsort(sig_f[peak_idx])
    sort_idx = sort_idx[::-1]

    peak_idx1 = peak_idx[sort_idx[0]]
    peak_idx2 = peak_idx[sort_idx[1]]

    f_hr1 = peak_idx1 / sig.shape[0] * fs
    hr1 = f_hr1 * 60

    f_hr2 = peak_idx2 / sig.shape[0] * fs
    hr2 = f_hr2 * 60
    if harmonics_removal:
        if np.abs(hr1 - 2 * hr2) < 10:
            hr = hr2
        else:
            hr = hr1
    else:
        hr = hr1

    x_hr = np.arange(len(sig)) / len(sig) * fs * 60
    return hr, sig_f_original, x_hr


def my_peak_detect(sig, samplingrate):
    bvp_detrend = detrend(sig, type='linear')  # useless...

    # bvp_detrend = polynomial(bvp, order=10)
    # # visualize detrend.
    # plt.figure()
    # plt.plot(bvp)
    # plt.plot(bvp_detrend)

    height = np.mean(bvp_detrend)  # minimal required height.
    distance = samplingrate / 2  # Required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
    peaks = find_peaks(bvp_detrend, height=height, distance=distance)[0]

    hr_list = []
    if len(peaks) <= 1:  # during training, we only expect 2 peaks at least.
        hr = 0
    else:
        for i in range(len(peaks) - 1):
            hr = 60 * samplingrate / (peaks[i + 1] - peaks[i])
            hr_list.append(hr)
        hr = np.mean(np.array(hr_list))
    return hr, peaks

def my_hr_cal(tmp, samplingrate=30):
    """
    MM论文 SSM的反推代码 find peak
    :param tmp:
    :param samplingrate:
    :return:
    """
    f1 = 0.8
    f2 = 2.8
    samplingrate = samplingrate
    b, a = signal.butter(4, [2 * f1 / samplingrate, 2 * f2 / samplingrate], 'bandpass')
    tmp = signal.filtfilt(b, a, np.array(tmp))
    # tmp = cwt_filtering(tmp, samplingrate)[0]

    hr_caled, peaks = my_peak_detect(tmp, samplingrate)
    return hr_caled, tmp, peaks

def my_hr_cal_batch(tmp, samplingrate=30, ssm_flag = 1, mode = 'peak'):
    hr_caled = []
    if ssm_flag == 1:
        for i in tmp:
            ssm = my_self_similarity_calc(i, 30)
            diag = []
            for j in range(0, len(ssm)):
                diag.append(np.diagonal(ssm, offset=j).mean())  # 获取斜对角
            diag = np.array(diag)
            if mode=='peak':
                hr, tmp, peaks = my_hr_cal(diag)
            else:
                hr,_,_ = hr_fft(diag)

            # hr,_ = my_hr_cal(i, samplingrate=samplingrate)
            hr_caled.append(hr)
    else:
        for i in tmp:
            if mode =='peak':
                hr, tmp, peaks = my_hr_cal(i)
            else:
                hr,_,_ = hr_fft(i)

            # hr,_ = my_hr_cal(i, samplingrate=samplingrate)
            hr_caled.append(hr)
    return np.array(hr_caled)


if __name__ == '__main__':
    x1 = np.load("/data/xieyiping/projectone/Physnet/Wloss/result_PURE_WD/1/2/02-01.npy", allow_pickle=True).item()
    # print(len(x1['rppg_list']))
    rppg0 = x1['rppg_list'][0]
    # print(len(rppg0))

    # ssm = self_similarity_calc(rppg0)
    ssm1 = my_self_similarity_calc(rppg0, 11)
    print(ssm1.shape)
    # print(type(ssm))
    # print(type(ssm[0]))
    # print(type(ssm1))
    # print(type(ssm1[0]))

    # print(len(ssm[0]))
    # print(len(ssm))

    diag = []
    for i in range(0,len(ssm1)):
        diag.append(np.diagonal(ssm1, offset=i).mean())  # 获取斜对角
    print(len(diag))

    print(diag)



