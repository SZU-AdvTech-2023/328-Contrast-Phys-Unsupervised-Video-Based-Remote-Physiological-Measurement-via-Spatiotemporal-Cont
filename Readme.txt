This project used contrast-phys to conduct training tests on the PURE, COHFACE, and UBFC datasets, and to design a new Loss (see loss/myloss.py) and a new SSM module (SSM.py).
reffered code: https://github.com/zhaodongsun/contrast-phys

—————————————————train and test———————————————————
Train:
The train files (train_PURE, train_UBFC, train_COHFACE ) are used to test training on the COHFACE, PURE, and UBFC datasets, respectively.

Test_all:
The test_all files (COHFACE_test_all.py,   PURE_test_all.py,  PURE_test_all.py ) are for tests that do all epoch results on the COHFACE, PURE, UBFC datasets；

Test:
The test files (tese_PURE, test_UBFC, test_COHFACE) are used to test for a certain epoch on the COHFACE, PURE, UBFC datasets.

Others:
SSM.py    is a new module added to this article - the self-similarity map module
fig_show   is a file that graphically displays the bvp signal saved by the test
vedio_show   is a file with a video showing the heart rate and its changing waveforms

——————————————————Model———————————————————————
Contains two files：
IrrelevantPowerRatio:  process the irregular part of the obtained signal bvp
PhysNetModel:  overall modeling framework


——————————————————loss———————————————————————
Contains the original loss function (loss), which uses only numerical alignment;
The improved loss function (myloss), which uses distributional JSP alignment.

——————————————————utils———————————————————————
The utils_/_npz (utils_cohface_npz, utils_pure_npz, utils_ubfc_npz) separate dataloaders for reading different data sets, etc.
utils_sig:  Some ways to post-process the signal


——————————————————data processing———————————————————
COHFACE_data_preprocess, PURE-rppg_data_process, UBFC-rPPG_data_preprocess  contains the preprocessing approach of this paper for the three datasets

—————————————————————————————————————————————
If you want to apply the code in this article, you need to follow these steps:
① Preprocess your dataset, refer to the documentation in /data processing;
   **You need to change your own dataset location in the file**

② Train your code, train the train_PURE, train_UBFC   or   train_COHFACE .
   **You need to change your own dataset location in the file**

③If you want to test a single saved file, use test, if you want to test all epochs of files, use the corresponding test_all
   **You need to change your own dataset location in the file**

④If you want to plot to show the bvp signal, run fig_show; if you want to show the video heart rate and its waveform, use vedio_show
   **You need to change your own dataset location in the file**




