#!/opt/conda/bin/python
import datetime
import os
import time

import cv2
import face_alignment
import h5py
import numpy as np
import pandas as pd
# from mtcnn import MTCNN
from facenet_pytorch import MTCNN
from tqdm import tqdm
from shutil import copyfile

def modify_video_frame_rate(vid_path,destFps,des_video_path):
    # dir_name = os.path.dirname(vid_path)
    # basename = os.path.basename(videoPath)
    # video_name = basename[:basename.rfind('.')]
    # video_name = video_name + "moify_fps_rate"
    # resultVideoPath = f'{dir_name}/{video_name}.mp4'

    videoCapture = cv2.VideoCapture(vid_path)

    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    if fps != destFps:
        frameSize = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # 这里的VideoWriter_fourcc需要多测试，如果编码器不对则会提示报错，根据报错信息修改编码器即可
        videoWriter = cv2.VideoWriter(des_video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), destFps, frameSize)

        i = 0;
        while True:
            success, frame = videoCapture.read()
            if success:
                i += 1
                # print('转换到第%d帧' % i)
                videoWriter.write(frame)
            else:
                print(vid_path,'帧率转换结束')
                break


# modify_video_frame_rate("D:\\DataSet\\RppgDataset2\\cohface\\1\\0\\data.avi",30,"D:\\DataSet\\RppgDataset2\\COHFACE_30hz\\1\\0\\data.avi")
# copyfile("D:\\DataSet\\RppgDataset2\\cohface\\1\\0\\data.hdf5", "D:\\DataSet\\RppgDataset2\\COHFACE_30hz\\1\\0\\data.hdf5")

def modify_COHFACE_frame_rate(root_path,destFps,des_path):

    subjects = os.listdir(root_path)  # subject1, subject 10
    subjects.sort()
    # subjects1=subjects1[:1]

    for subject in tqdm(subjects):  # subject1
        subject_path = os.path.join(root_path, subject)  # '/data1/vsign/COHFACE/COHFACE/1'
        if (os.path.isdir(subject_path)) & (subject != 'protocols'):
            scenarios = os.listdir(subject_path)
            scenarios.sort()
            for sce in scenarios:
                sce_path = os.path.join(subject_path, sce)  # '/data1/vsign/COHFACE/COHFACE/1/0'
                if not os.path.exists(os.path.join(os.path.join(des_path, subject), sce)):
                    os.makedirs(os.path.join(os.path.join(des_path, subject), sce))
                des_video_path = os.path.join(os.path.join(des_path, subject), sce)  # '/data1/vsign/COHFACE_moify_fps/1/0'
                files = os.listdir(sce_path)
                for file in files:  # vid.avi
                    if file.endswith('avi'):
                        vid_path = os.path.join(sce_path, file)  # '/data1/vsign/COHFACE/COHFACE/1/0/data.avi'
                        des_vid_path = os.path.join(des_video_path, file)
                        modify_video_frame_rate(vid_path,destFps,des_vid_path)
                    else:
                        copyfile(os.path.join(sce_path, file), os.path.join(des_video_path, file))



# root_path = "D:\\DataSet\\RppgDataset2\\cohface"
# des_path = "D:\\DataSet\\RppgDataset2\\COHFACE_30hz"
# modify_COHFACE_frame_rate(root_path,30,des_path)

# import pyedflib
def IsInteger(text):
    try:
        int(text)
        return True
    except ValueError:
        return False

def extract_video_frame(video_path, des_path):
    """Extract frames from the given video

    Extract each frame from the given video file and store them into '.jpg' format. It
    extracts every frame of the video. If the given frame path exsits, it overwrites
    the contents if users choose that.

    Args:
            video_path (str): Required. The path of video file.

            frame_path (str): Required. The path to store extracted frames. If the path exists, it tries to
                                    remove it by asking the user.

    Raises:
            OSError: If the given video path is incorrect, or the video cannot be opened by
                            Opencv.
            ValueError: If the given specified range out of range
    """

    frames = []
    count = 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError("Failed to open input video")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if not os.path.exists(des_path):
        os.makedirs(des_path)

    for frameId in range(int(frame_count)-1):
        ret, frame = cap.read()

        img_name = "image_{:05d}.jpg".format(frameId)
        img_path= os.path.join(des_path, img_name)

        ret = cv2.imwrite(img_path, frame)
        count += 1

    cap.release()
    return frames


def video_to_images_COHFACE(data_path,des_path):

    start = time.time()
    subjects = os.listdir(data_path) # subject1, subject 10
    subjects.sort()
    # subjects1=subjects1[:1]

    for subject in tqdm(subjects): # subject1
        if not IsInteger(subject):
            continue
        subject_path = os.path.join(data_path, subject) # '/data1/vsign/COHFACE_filter/COHFACE/subject1'
        out_subject_path = os.path.join(des_path, subject)
        if not os.path.exists(out_subject_path):
            os.makedirs(out_subject_path)

        for sample_id in os.listdir(subject_path):
            input_sample_path = os.path.join(subject_path,sample_id)
            img_path = os.path.join(out_subject_path,sample_id) # '/data1/vsign/COHFACE_filter_imgs/subject1'
            files = os.listdir(input_sample_path)
            for file in files:  # vid.avi
                if file.endswith('avi'):
                    vid_path = os.path.join(input_sample_path, file) # '/data1/vsign/COHFACE_filter/COHFACE/subject1/vid.avi'
                    extract_video_frame(vid_path, img_path)

    duration = str(datetime.timedelta(seconds=time.time() - start))
    print("It takes {} time for extracting COHFACE dataset".format(duration))


# data_path = 'D:\\DataSet\\RppgDataset2\\COHFACE_30hz'
# des_path = 'D:\\DataSet\\RppgDataset2\\cohface_after\\COHFACE_filter_imgs'
# video_to_images_COHFACE(data_path,des_path)



def check_imgs_len(des_path):
    subjects = os.listdir(des_path)  # subject1, subject 10
    subjects.sort()
    len_imgs = []

    for subject in tqdm(subjects):  # subject1
        subject_path = os.path.join(des_path, subject)  # '/data1/vsign/COHFACE_filter/COHFACE/subject1'
        imgs = os.listdir(subject_path)  # 1546 30fps
        len_img = len(imgs)
        len_imgs.append(len_img)
    return len_imgs

# des_path = '/data1/vsign/COHFACE_filter_imgs'
# lens=check_imgs_len(des_path) # frame len range: 1367-2051,fps=30,time_duration range:45.56s-68.36s



def save_landmarks(data_path,des_path):
    # data_path = '/data1/vsign/COHFACE_filter_imgs'
    # des_path='/data1/vsign/COHFACE_filter_landmarks'

    start = time.time()
    subjects = os.listdir(data_path)  # subject1, subject 10
    subjects.sort()
    # subjects1=subjects1[:1]
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    if not os.path.exists(des_path):
        os.makedirs(des_path)

    for subject in tqdm(subjects):  # subject1
        subject_path = os.path.join(data_path, subject)  # '/data1/vsign/COHFACE_filter_imgs/COHFACE/subject1'
        landmarks = fa.get_landmarks_from_directory(subject_path)  # dict 1546, each list 2(detected two faces) or 1
        landmark_keys = list(landmarks.keys())
        landmark_values = [d[0] for d in list(landmarks.values())]

        # new_dict = {}
        # for i in range(len(landmark_keys)):
        #     new_dict[landmark_keys[i]] = landmark_values[i]
        # file_path=des_path+'/'+subject+'.csv' # '/data1/vsign/COHFACE_filter_landmarks/subject1.csv'
        # with open(file_path, 'w') as f:
        #     for key in new_dict.keys():
        #         f.write("%s,%s\n" % (key, new_dict[key]))

        file_path = os.path.join(des_path, subject)
        np.savez(file_path, landmark_keys=landmark_keys, landmark_values=landmark_values)

    duration = str(datetime.timedelta(seconds=time.time() - start))
    print("It takes {} time for extracting COHFACE dataset".format(duration))

# data_path = '/data1/vsign/COHFACE_filter_imgs'
# des_path='/data1/vsign/COHFACE_filter_landmarks'
# save_landmarks(data_path,des_path)


# file_path='/data1/vsign/COHFACE_filter_landmarks/subject1.npz'
# file=np.load(file_path)
# landmark_keys=file['landmark_keys']  # array 1546, each frame
# landmark_values=file['landmark_values'] # array (1546,68,2), landmarks for each ROI on faces, face area is smaller


def crop_face_img(img_path,deth_path,bounding_box,crop_shape=(128,128)):
    # print(img_path)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) # array
    # # mtcnn = MTCNN(device='cuda:0')
    # mtcnn = MTCNN(device='cpu:0')
    #
    # result= mtcnn.detect(img) # pytorch return  (boxes, probs) tensorflow return list[]
    # h=img.shape[0] # 480
    # w=img.shape[1] # 640
    #
    # # list [{'box': [110, 89, 235, 314],(x,y,width,height) 'confidence': 0.9999747276306152,
    # # 'keypoints': {'left_eye': (181, 215), 'right_eye': (286, 218), 'nose': (235, 286), 'mouth_left': (190, 335), 'mouth_right': (273, 338)}}]
    #
    # # if len(result[0][0])==0:
    # if result[0] is None:
    #     # If landmarks cannot be detected, return a bbx of first frame
    #     bounding_box= [0,0,w,h] # bounding_box [wmin,hmin,wmax,hmax],
    #
    # else:
    #     bounding_box = result[0][0] # bounding_box [wmin,hmin,wmax,hmax],[[286.418   151.44427 431.32965 348.65497]]

    crop_img = img[max(0,int(bounding_box[1])):max(0,int(bounding_box[3])), max(0,int(bounding_box[0])):max(0,int(bounding_box[2]))] # (224,172,3) crop_img=img[hmin:hmax,wmin:wmax]

    # resized_crop_img=cv2.resize(crop_img,crop_shape) #(128,128,3)

    # plt.figure()
    # plt.imshow(resized_crop_img)
    # plt.show()

    cv2.imwrite(deth_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

# img_path='test1.png'
# deth_path='ok1.png'
# crop_face_img(img_path,deth_path)
def get_bounding_box(img_path):
    # print(img_path)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)  # array
    # mtcnn = MTCNN(device='cuda:0')
    mtcnn = MTCNN(device='cpu:0')

    result = mtcnn.detect(img)  # pytorch return  (boxes, probs) tensorflow return list[]
    h = img.shape[0]  # 480
    w = img.shape[1]  # 640

    # list [{'box': [110, 89, 235, 314],(x,y,width,height) 'confidence': 0.9999747276306152,
    # 'keypoints': {'left_eye': (181, 215), 'right_eye': (286, 218), 'nose': (235, 286), 'mouth_left': (190, 335), 'mouth_right': (273, 338)}}]

    # if len(result[0][0])==0:
    if result[0] is None:
        # If landmarks cannot be detected, return a bbx of first frame
        bounding_box = [0, 0, w, h]  # bounding_box [wmin,hmin,wmax,hmax],

    else:
        bounding_box = result[0][0]  # bounding_box [wmin,hmin,wmax,hmax],[[286.418   151.44427 431.32965 348.65497]]

    return bounding_box

def crop_images_COHFACE_to_face(data_path,des_path):

    start = time.time()
    subjects = os.listdir(data_path) # subject1,subject2
    subjects.sort() #
    try:
        for subject in tqdm(subjects):
            subject_path = os.path.join(data_path, subject) #
            for sample_id in os.listdir(subject_path):
                sample_path = os.path.join(subject_path,sample_id)

                if os.path.isdir(sample_path):
                    imgs = os.listdir(sample_path)  #
                    imgs.sort()
                    des_sub_path = os.path.join(des_path, subject, sample_id) #
                    flag = 0
                    for img in imgs:  # img.png
                        img_path = os.path.join(sample_path,
                                                img)  # '/data1/vsign/COHFACE_filter_imgs/subject1/image_00000.jpg'
                        crop_face_path = os.path.join(des_sub_path,
                                                      img)  # '/data1/vsign/COHFACE_filter_crop_imgs/subject1/image_00000.jpg'
                        if flag == 0:
                            bounding_box = get_bounding_box(img_path)
                            flag = 1

                        # print(img_path)
                        if not os.path.exists(des_sub_path):
                            os.makedirs(des_sub_path)
                        crop_face_img(img_path, crop_face_path,bounding_box)

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        print(e, "\n")

    duration = str(datetime.timedelta(seconds=time.time() - start))
    print("It takes {} time for extracting PURE dataset".format(duration))

# img_path = 'D:\\DataSet\\RppgDataset2\\cohface_after\\COHFACE_filter_imgs'
# deth_path = 'D:\\DataSet\\RppgDataset2\\cohface_after\\COHFACE_filter_crop_imgs'
# crop_images_COHFACE_to_face(img_path,deth_path)

# subjects=os.listdir('/data1/vsign/COHFACE_filter/COHFACE') #42, COHFACE_rPPG
# imgs_array=cv2.imread('/data1/vsign/COHFACE_filter_imgs/subject1/image_00000.jpg') #(480,640,3)

def COHFACE_img_to_numpy(root_path,des_path):

    if not os.path.exists(des_path):
        os.makedirs(des_path)

    start = time.time()
    subjects = os.listdir(root_path)  # 1,2,...10...
    subjects.sort()  #

    try:
        for subject in tqdm(subjects):  # 1,2
            subject_path = os.path.join(root_path, subject)  # '/data1/vsign/VIPL_v2_crop_imgs/1'
            out_subject_path = os.path.join(des_path, subject)
            if not os.path.exists(out_subject_path):
                os.makedirs(out_subject_path)
            for sample_id in os.listdir(subject_path):
                sample_path = os.path.join(subject_path, sample_id)

                imgs = os.listdir(sample_path)  #
                imgs.sort()
                des_sub_path = os.path.join(out_subject_path, sample_id)# '/data1/vsign/COHFACE_filter_crop_numpy/subject1'
                img_list=[]
                for img in imgs:
                    img_path = os.path.join(sample_path, img)  # '/data1/vsign/COHFACE_filter_crop_imgs/subject1/image_00000.jpg'
                    img = cv2.imread(img_path) # (173,142,3)
                    resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                    img_list.append(resized)
                np.savez(des_sub_path, frame=np.array(img_list))

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        print(e, "\n")

    duration = str(datetime.timedelta(seconds=time.time() - start))
    print("It takes {} time for extracting PURE dataset".format(duration))


# root_path='D:\\DataSet\\RppgDataset2\\cohface_after\\COHFACE_filter_crop_imgs'
# des_path='D:\\DataSet\\RppgDataset2\\cohface_after\\COHFACE_filter_crop_numpy'
# COHFACE_img_to_numpy(root_path,des_path)


# VIPL-HR-V1_cropped_numpy,PURE_crop_numpy,VIPL_v2_crop_numpy saved in cv2 format


# frames_path='/data1/vsign/VIPL_v2_crop_numpy/sub100_video1.npz'
# frames=np.load(frames_path)['frame'] # (265,128,128,3)
# img1=cv2.cvtColor(frames[0],cv2.COLOR_BGR2RGB)
# # cv2.imwrite('/data1/vsign/test_img.jpg',img1) blu, cv2 imwrite must convert to BGR
# plt.figure()
# plt.imshow(img1)
# plt.show()


def convert_timestamps(current_timestamps, first_timestamp):
    return (current_timestamps - first_timestamp)



def CorrectIrregularlySampledData(df, Fs): # Fs=30 frame per second, 1/Fs: sec per frame
    if df.iloc[0]['timestamp'] > 0.0:
        top_row = df.iloc[[0]].copy()
        df = pd.concat([top_row, df], ignore_index=True)
        df.loc[0, 'timestamp'] = 0.0
    new_data = []
    for frame_on, time_on in enumerate(np.arange(0.0, df.iloc[-1]['timestamp'], 1 / Fs)): #0,1/30,2*1/30,3*1/30...,final timestamp, for every resample rate
        time_diff = (df['timestamp'] - time_on).to_numpy() # array of time difference of each signal to current signal, negative for previous signal
        stop_idx = np.argmax(time_diff > 0) # choose the first value that >0, current frame, start from second frame (id=1),15437
        start_idx = stop_idx - 1 # 15436
        time_span = time_diff[stop_idx] - time_diff[start_idx] # time span for two signals,not frames, 0.003906
        rel_time = -time_diff[start_idx] #0.00312
        stop_weight = rel_time / time_span #0.7999
        start_weight = 1 - stop_weight  # 0.2
        average_row = pd.concat([df.iloc[[start_idx]].copy() * start_weight, df.iloc[[stop_idx]].copy() * stop_weight]).sum().to_frame().T

        # df_startidx=df.iloc[[start_idx]].copy() # row of start_idx,df (1,3) timestamp,ppg,respiration,(60.29,36,08,20.95)
        # average_row1 = pd.concat(
        #     [df.iloc[[start_idx]].copy() * start_weight, df.iloc[[stop_idx]].copy() * stop_weight])
        # average_row2=average_row1.sum() #series,(60.3,35,7,20.95)
        # average_row3=average_row2.to_frame().T
        new_data.append(average_row)
    return pd.concat(new_data) # original data is 15440, resample new_data is 1810

def IsInteger(text):
    try:
        int(text)
        return True
    except ValueError:
        return False

def COHFACE_meta(root_path,des_path):
    # 只对txt进行处理
    if not os.path.exists(des_path):
        os.makedirs(des_path)

    start = time.time()
    subjects = os.listdir(root_path)  # 1,2,...10...
    subjects.sort()  #

    # try:
    for subject in tqdm(subjects):  # 1,2
        if not IsInteger(subject):
            continue
        subject_path = os.path.join(root_path, subject)  #
        out_subject_path = os.path.join(des_path, subject)
        # print(subject_path)
        if not os.path.exists(out_subject_path):
            os.makedirs(out_subject_path)
        for sample_id in os.listdir(subject_path):
            sample_path = os.path.join(subject_path,sample_id)

            files = os.listdir(sample_path)  # video, hdf5
            for file in files:
                if file.endswith('hdf5'):
                    file_path = os.path.join(sample_path, file)  # '"D:\DataSet\RppgDataset2\cohface\1\0\data.hdf5"'

                    with h5py.File(file_path, 'r') as hdf5_file:
                        # Write the phys.csv file.
                        Fs = float(hdf5_file.attrs['sample-rate-hz'][0])  # 256.0
                        phys_data = {}
                        phys_data['timestamp'] = hdf5_file['time'][:]
                        phys_data['timestamp'] -= phys_data['timestamp'][0]
                        phys_data['ppg'] = hdf5_file['pulse'][:]
                        phys_data['respiration'] = hdf5_file['respiration'][:]
                        df = pd.DataFrame(phys_data)
                        # df = CorrectIrregularlySampledData(df, 30.0)
                        df = CorrectIrregularlySampledData(df, 30.0) # COHFACE  video 20Hz->30
                    # raw_data = []
                    # with open(file_path, "r") as phys_file:
                    #     for line in phys_file:
                    #         row = line.split()
                    #         row = [float(x) for x in row]  # list (1547)
                    #         raw_data.append(np.array(row))  # list 3,first is bvp, second is hr,third is timestamp
                    #
                    # first_timestamp = raw_data[2][0]
                    # phys_data = {}
                    # phys_data['timestamp'] = convert_timestamps(raw_data[2], first_timestamp) # 获取时间戳
                    # phys_data['ppg'] = raw_data[0] # array 1547
                    # phys_data['ppg_heart_rate'] = raw_data[1]
                    # df = pd.DataFrame(phys_data)
                    # df = CorrectIrregularlySampledData(df, 30.0) # (1547,3)

                    des_file_path = os.path.join(des_path, subject, sample_id)  #'/data1/vsign/COHFACE_filter_meta/subject1'
                    df.to_excel(des_file_path + '.xlsx')
                    np.savez(des_file_path, respiration=df['respiration'].values,wave=df['ppg'].values, frame_time=df['timestamp'].values, fps_cal=30)

    # except (KeyboardInterrupt, SystemExit):
    #     raise
    # except Exception as e:
    #     print(e, "\n")

    duration = str(datetime.timedelta(seconds=time.time() - start))
    print("It takes {} time for extracting COHFACE dataset".format(duration))

root_path='D:\\DataSet\\RppgDataset2\\COHFACE_30hz'
des_path='D:\\DataSet\\RppgDataset2\\cohface_after\\COHFACE_filter_meta_numpy'
COHFACE_meta(root_path,des_path)


