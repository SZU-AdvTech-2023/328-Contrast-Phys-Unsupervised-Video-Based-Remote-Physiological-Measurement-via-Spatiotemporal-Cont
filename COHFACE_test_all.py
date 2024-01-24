import numpy as np
import torch
from PhysNetModel import PhysNet
from utils_data import *
from utils_sig import *
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import json
from SSM import my_hr_cal_batch

ex = Experiment('model_pred', save_git_info=False)


@ex.config
def my_config():
    # test iner-data
    epo = 39  # the model checkpoint at epoch e
    train_exp_num = 3  # the training experiment number
    train_exp_dir = '/data/xieyiping/projectone/Physnet/Contrast-phys/result_COHFACE/%d' % train_exp_num  # training experiment directory
    # # test cross-data
    # epo = 29  # the model checkpoint at epoch e
    # train_exp_num = 1  # the training experiment number
    # train_exp_dir = '/data/xieyiping/projectone/Physnet/Contrast-phys/result_PURE/%d' % train_exp_num  # training experiment directory

    # train_exp_res = '/data/xieyiping/projectone/Physnet/Contrast-phys/result'
    time_interval = 30  # get rppg for 30s video clips, too long clips might cause out of memory

    ex.observers.append(FileStorageObserver(train_exp_dir))
    # device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    else:
        device = torch.device('cpu')


@ex.automain
def my_main(_run, epo, train_exp_dir, device, time_interval):
    # load test file paths
    test_list = list(np.load(train_exp_dir + '/test_list.npy'))
    pred_exp_dir = train_exp_dir + '/%d' % (int(_run._id))  # prediction experiment directory

    @torch.no_grad()
    def dl_model(imgs_clip):
        # model inference
        img_batch = imgs_clip
        img_batch = img_batch.transpose((3, 0, 1, 2))
        img_batch = img_batch[np.newaxis].astype('float32')
        img_batch = torch.tensor(img_batch).to(device)

        rppg = model(img_batch)[:, -1, :]
        rppg = rppg[0].detach().cpu().numpy()
        return rppg


    with open(train_exp_dir + '/config.json') as f:
        config_train = json.load(f)

    best_mae = 1000
    best_epo = 0
    for i in range(0,15):

        model = PhysNet(config_train['S'], config_train['in_ch']).to(device).eval()
        # model.load_state_dict(torch.load(train_exp_dir+'/epoch%d.pt'% (epo), map_location=device)) # load weights to the model
        train_exp_dir2 = '/data/xieyiping/projectone/Physnet/Contrast-phys/result_COHFACE/1'
        # train_exp_dir2 = '/data/xieyiping/projectone/Physnet/Wloss/result_PURE_WD/4'
        model.load_state_dict(
            torch.load(train_exp_dir2 + '/epoch%d.pt' % (i), map_location=device))  # load weights to the model

        bvp_all = []
        rppg_all = []
        for h5_path in test_list:

            h5_path = str(h5_path)
            f = np.load(h5_path)  # read imgs znp files
            # with h5py.File(h5_path, 'r') as f:
            imgs = f['frame']
            meta_data = h5_path.replace('COHFACE_filter_crop_numpy', 'COHFACE_filter_meta_numpy')
            f2 = np.load(meta_data)
            bvp = f2['wave']
            # bvppeak = f['bvp_peak']
            fs = config_train['fs']

            duration = np.min([imgs.shape[0], bvp.shape[0]]) / fs
            num_blocks = int(duration // time_interval)

            rppg_list = []
            bvp_list = []

            # bvppeak_list = []
            # print('clip in %d blocks'%num_blocks)
            for b in range(num_blocks):
                rppg_clip = dl_model(imgs[b * time_interval * fs:(b + 1) * time_interval * fs])
                rppg_list.append(rppg_clip)
                rppg_all.append(rppg_clip)

                bvp_list.append(bvp[b * time_interval * fs:(b + 1) * time_interval * fs])
                bvp_all.append(bvp[b * time_interval * fs:(b + 1) * time_interval * fs])
                # bvppeak_list.append(bvppeak[b*time_interval*fs:(b+1)*time_interval*fs])

        gt_hr = sig_out_hr_batch(bvp_all, 0.6, 4, 30, order=2)
        rppg_hr = sig_out_hr_batch(rppg_all, 0.6, 4, 30, order=2)
        # gt_hr = my_hr_cal_batch(bvp_all, 30, ssm_flag=1, mode='fft')
        # rppg_hr = my_hr_cal_batch(rppg_all, 30, ssm_flag=1, mode='fft')


        # print('gt_hr:  ', gt_hr)
        # print('rppg_hr:  ', rppg_hr)

        # culculate metrics
        criterion_MAE = mean_absolute_error
        test_mae = criterion_MAE(rppg_hr, gt_hr)
        test_RMSE = rmse(rppg_hr, gt_hr)
        # pearson_avg = np.mean(pearson_list)
        test_r, _ = pearsonr(gt_hr, rppg_hr)
        # Mean and Std
        diff = rppg_hr - gt_hr
        mean = np.mean(diff)
        test_std = np.std(diff)
        print('Epoch%d: test mae is %.4f, std is %.4f,RMSE is %.4f, r is %.4f' % (i, test_mae, test_std, test_RMSE, test_r))
        if test_mae < best_mae:
            best_mae = test_mae
            best_epo = i
    print('best mae is %.4f , best epo is %.4f' % (best_mae, best_epo))



