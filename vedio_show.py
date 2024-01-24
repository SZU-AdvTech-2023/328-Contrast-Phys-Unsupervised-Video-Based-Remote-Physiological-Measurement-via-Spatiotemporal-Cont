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

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

else:
    device = torch.device('cpu')

model = PhysNet(2, 3).to(device).eval()
model.load_state_dict(torch.load("/data/xieyiping/projectone/Physnet/Contrast-phys/result_PURE/2/epoch29.pt", map_location=device))


@torch.no_grad()
def dl_model(imgs_clip):
    # model inference
    img_batch = imgs_clip
    img_batch = img_batch.transpose((3,0,1,2))
    img_batch = img_batch[np.newaxis].astype('float32')
    img_batch = torch.tensor(img_batch).to(device)

    rppg = model(img_batch)[:,-1, :]
    rppg = rppg[0].detach().cpu().numpy()
    return rppg

h5_path = "/data/xieyiping/dataset/PURE_numpy/PURE_filter_crop_numpy_box/01-06.npz"
f = np.load(h5_path) # read imgs znp files
# with h5py.File(h5_path, 'r') as f:
imgs = f['frame']
length = imgs.shape[0]
time_interval = 300

meta_data = h5_path.replace('PURE_filter_crop_numpy_box','PURE_filter_meta_numpy')
f2 = np.load(meta_data)
bvp = f2['wave']
# bvppeak = f['bvp_peak']
fs = 30

rppg_list = []
bvp_list = []
# hr_gt = []
# hr = []
for i in range(225,length-225):
    print(i)
    rppg_clip = dl_model(imgs[i-225:i+225])
    rppg_list.append(rppg_clip)
    bvp_list.append(bvp[i-225:i+225])

gt_hr = sig_out_hr_batch(bvp_list, 0.6, 4, 30, order=2)
rppg_hr = sig_out_hr_batch(rppg_list, 0.6, 4, 30, order=2)

rppg_list = np.array(rppg_list)
bvp_list = np.array(bvp_list)
gt_hr = np.array(gt_hr)
rppg_hr = np.array(rppg_hr)
results = {'rppg_list': rppg_list, 'bvp_list': bvp_list, 'hr_list': rppg_hr, 'hr_gt': gt_hr}
pred_exp_dir = '/data/xieyiping/projectone/Physnet/Contrast-phys/homework'
np.save(pred_exp_dir+'/man_test', results)   # h5_path.split('/')[-1][:-4] = 'subject1'
