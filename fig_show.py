import matplotlib.pyplot as plt
from utils_sig import *
import numpy as np


def hr_fft(sig, fs, harmonics_removal=True):
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

    # x_hr = np.arange(len(sig))/len(sig)*fs*60
    return hr

def butter_bandpass_filter(sig, lowcut, highcut, fs, order=2):
    # butterworth bandpass filter
    # signals of a video

    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, sig)

    return np.array(y)

x1 = np.load("/data/xieyiping/projectone/Physnet/Contrast-phys/result/2/2/subject48.npy",allow_pickle=True).item()

# print(x1['rppg_list'][0])
# print(x1['bvp_list'][0])
rppg0 = (x1['rppg_list'][0] - np.mean(x1['rppg_list'][0])) / np.std(x1['rppg_list'][0])
rppg1 = butter_bandpass_filter(rppg0, 0.6, 4, 30, order=2)
rppg2 = (rppg1 - np.mean(rppg1)) / np.std(rppg1)
# hr = hr_fft(rppg, fs = 30)
# print(hr)
bvp0 = (x1['bvp_list'][0] - np.mean(x1['bvp_list'][0])) / np.std(x1['bvp_list'][0])
bvp1 = butter_bandpass_filter(bvp0, 0.6, 4, 30, order=2)
bvp2 = (bvp1 - np.mean(bvp1)) / np.std(bvp1)
# gt_hr = hr_fft(bvp, fs = 30)
# print(gt_hr)


# print(rppg2)
# print(bvp2)

# # yr = (yr - np.mean(yr)) / np.std(yr)
# bvp = butter_bandpass_filter(x1['bvp_list'][0], 0.6, 4, 30, order=2)
#
plt.subplots_adjust(right=0.7)
plt.plot(rppg0, alpha=0.7, label='outputs\n rppg')
plt.plot(rppg2, label='outputs\n filter_rppg') # 滤波后的结果
plt.plot(bvp2, '--', label='referencja\n PPG')
#
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='large')
plt.ylabel('Amplituda', fontsize='large', fontweight='semibold')
plt.xlabel('Time [sample]', fontsize='large', fontweight='semibold')
plt.grid()
plt.xlim([350, 550])
plt.ylim([-2, 3])
#
# #     plt.savefig('3d.svg', bbox_inches='tight')
plt.show()