import numpy as np
import os
import h5py
from torch.utils.data import Dataset
# from scipy.fft import fft
# from scipy import signal
# from scipy.signal import butter, filtfilt

def COHFACE_LU_split():
    # split PURE dataset into training and testing parts
    # the function returns the file paths for the training set and test set.
    # TODO: if you want to train on another dataset, you should define new train-test split function.

    h5_dir = '/data/xieyiping/dataset/COHFACE/COHFACE_filter_crop_numpy/'
    train_list = []
    val_list = []

    val_subject = [3,8,10,11,12,13,14,15,20,22,23,26,30,32,34,40]
    # clean = [1,2]

    for subject in range(1, 41):
        subject_path = os.path.join(h5_dir, str(subject))
        if subject in val_subject:
            for i in os.listdir(subject_path):
                # if int(i[0]) in clean:
                val_list.append(subject_path + '/'+i)
        else:
            for i in os.listdir(subject_path):
                # if int(i[0]) in clean:
                train_list.append(subject_path + '/'+i)

    return train_list, val_list

# train_list, val_list = COHFACE_LU_split()
# print(train_list)
# print(val_list)
# print(len(val_list))  #64
# print(len(train_list))  # 100


class COHFACEdataset(Dataset):

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
