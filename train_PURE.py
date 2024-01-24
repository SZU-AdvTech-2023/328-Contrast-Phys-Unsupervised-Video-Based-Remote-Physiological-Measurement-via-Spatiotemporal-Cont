# import matplotlib.pyplot as plt
import torch
from PhysNetModel import PhysNet
# from loss import ContrastLoss
from myloss import ContrastLoss
from IrrelevantPowerRatio import IrrelevantPowerRatio

from utils_sig import *
from torch import optim
from torch.utils.data import DataLoader
from sacred import Experiment
from sacred.observers import FileStorageObserver
from utils_pure_npz import *

ex = Experiment('model_train', save_git_info=False)

if torch.cuda.is_available():
    device = torch.device('cuda:3')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')


@ex.config
def my_config():
    # here are some hyperparameters in our method
    _run = 1
    # hyperparams for model training
    total_epoch = 30  # total number of epochs for training the model
    lr = 1e-5  # learning rate
    in_ch = 3  # TODO: number of input video channels, in_ch=3 for RGB videos, in_ch=1 for NIR videos.

    # hyperparams for ST-rPPG block
    fs = 30  # video frame rate, TODO: modify it if your video frame rate is not 30 fps.
    T = fs * 10  # temporal dimension of ST-rPPG block, default is 10 seconds.
    S = 2  # spatial dimenion of ST-rPPG block, default is 2x2.

    # hyperparams for rPPG spatiotemporal sampling
    delta_t = int(T / 2)  # time length of each rPPG sample
    K = 4  # the number of rPPG samples at each spatial position

    result_dir = '/data/xieyiping/projectone/Physnet/Contrast-phys/result_PURE'  # TODO:store checkpoints and training recording
    ex.observers.append(FileStorageObserver(result_dir))
    # my_main()


@ex.automain
def my_main(_run, total_epoch, fs, T, S, lr, in_ch, result_dir, delta_t, K):
    # total_epoch = 30   # train epoch
    # fs = 30   # video frame rate
    # T = fs*10   # temporal dimension of ST-rPPG block
    # S = 2   # spatial dimenion of ST-rPPG block
    # lr = 1e-5   # learning rate
    # in_ch = 3  # the channel
    # result_dir = '/data/xieyiping/projectone/Physnet/Contrast-phys/result'
    #
    # delta_t = int(T/2)  # time length of each rPPG sample
    # K = 4  # the number of rPPG samples at each spatial position

    exp_dir = result_dir + '/%d' % (int(_run._id))  # store experiment recording to the path

    # get the training and test file path list by spliting the dataset
    train_list, test_list = PURE_LU_split()  # TODO: you should define your function to split your dataset for training and testing
    np.save(exp_dir + '/train_list.npy', train_list)
    np.save(exp_dir + '/test_list.npy', test_list)

    # define the dataloader
    # 训练数据中没有标签
    dataset = PUREdataset(train_list, T)  # please read the code about H5Dataset when preparing your dataset
    dataloader = DataLoader(dataset, batch_size=2,  # two videos for contrastive learning
                            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # define the model and loss
    model = PhysNet(S, in_ch=in_ch).to(device).train()
    loss_func = ContrastLoss(delta_t, K, fs, high_pass=40, low_pass=250)

    # define irrelevant power ratio
    IPR = IrrelevantPowerRatio(Fs=fs, high_pass=40, low_pass=250)

    # define the optimizer
    opt = optim.AdamW(model.parameters(), lr=lr)

    loss_now = 100
    loss = 100
    for e in range(total_epoch):
        print('epoch%d start' % e)
        for it in range(np.round(67 / (T / fs)).astype('int')):
            # TODO: 60 means the video length of each video is 60s. If each video's length in your dataset is other value (e.g, 30s), you should use that value.
            print('epoch{} it{} start:'.format(e, it))
            for imgs in dataloader:  # dataloader randomly samples a video clip with length T
                imgs = imgs.to(device)

                # model forward propagation
                model_output = model(imgs)
                rppg = model_output[:, -1]  # get rppg

                # define the loss functions
                loss, p_loss, n_loss = loss_func(model_output)

                # optimize
                opt.zero_grad()
                loss.backward()
                opt.step()

                # evaluate irrelevant power ratio during training
                ipr = torch.mean(IPR(rppg.clone().detach()))

                # save loss values and IPR
                ex.log_scalar("loss", loss.item())
                ex.log_scalar("p_loss", p_loss.item())
                ex.log_scalar("n_loss", n_loss.item())
                ex.log_scalar("ipr", ipr.item())
        print('epoch%d loss:', loss.item())
        # save model checkpoints
        # if loss < loss_now:
        #     loss_now = loss
        #     print('save epoch%d model' % e)
        torch.save(model.state_dict(), exp_dir + '/epoch%d.pt' % e)

# if __name__ == '__main__':
#     my_main()
