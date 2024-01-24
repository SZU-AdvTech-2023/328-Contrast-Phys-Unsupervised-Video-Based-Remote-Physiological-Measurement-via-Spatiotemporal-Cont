import numpy as np
import os
import h5py
from torch.utils.data import Dataset
# from scipy.fft import fft
# from scipy import signal
# from scipy.signal import butter, filtfilt

def UBFC_LU_split():
    # split UBFC dataset into training and testing parts
    # the function returns the file paths for the training set and test set.
    # TODO: if you want to train on another dataset, you should define new train-test split function.

    h5_dir = '/data/xieyiping/dataset/UBFC_numpy/UBFC_filter_crop_numpy_box'
    train_list = []
    val_list = []

    val_subject = [49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38]

    for subject in range(1, 50):
        # print(os.path.isfile(h5_dir + '/subject%d.npz' % (subject)))
        if os.path.isfile(h5_dir + '/subject%d.npz' % (subject)):
            if subject in val_subject:
                val_list.append(h5_dir + '/subject%d.npz' % (subject))
            else:
                train_list.append(h5_dir + '/subject%d.npz' % (subject))

    return train_list, val_list

# train_list, val_list = UBFC_LU_split()
# print(train_list)
# print(val_list)
# print(len(val_list) + len(train_list))  #38


class UBFCdataset(Dataset):

    def __init__(self, train_list, T):
        self.train_list = train_list  # list of .h5 file paths for training
        self.T = T # video clip length

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        # get data from npz files, train data do not need label
        imgs = np.load(self.train_list[idx])

        img_length = imgs['frame'].shape[0]

        idx_start = np.random.choice(img_length - self.T)
        idx_end = idx_start + self.T

        img_seq = imgs['frame'][idx_start:idx_end]
        img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype(('float32'))

        return img_seq

        # get data from h5 files
        # with h5py.File(self.train_list[idx], 'r') as f:
        #     img_length = f['imgs'].shape[0]
        #
        #     idx_start = np.random.choice(img_length-self.T)
        #
        #     idx_end = idx_start+self.T
        #
        #     img_seq = f['imgs'][idx_start:idx_end]
        #     img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
        # return img_seq