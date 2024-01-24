"""
读取 ppg 信号
01-01: images
json:  array 2 x T bvp; hr
"""
import datetime
import os
import time
from scipy import interpolate
import cv2
import face_alignment
import numpy as np
import pandas as pd
# from mtcnn import MTCNN
from facenet_pytorch import MTCNN
from tqdm import tqdm
import json

def find_box(image):
    mtcnn = MTCNN()
    # 检测人脸并获取边界框
    boxes, _ = mtcnn.detect(image)
    box = boxes[0]

    # 计算边界框的高度
    vertical_range = box[3] - box[1]

    # 计算矩形框的大小并进行扩展
    scale = 1.2
    bounding_box_size = int(vertical_range * scale)
    box_center_x, box_center_y = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    box[0], box[1] = int(box_center_x - bounding_box_size // 2), int(box_center_y - bounding_box_size // 2)
    box[2], box[3] = int(box_center_x + bounding_box_size // 2), int(box_center_y + bounding_box_size // 2)

    # 确保边界框不超出图像
    h, w = image.shape[:2]
    box[0] = max(0, box[0])
    box[1] = max(0, box[1])
    box[2] = min(w, box[2])
    box[3] = min(h, box[3])

    # 裁剪图像
    # cropped_image = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    # box2 = [int(box[i]) for i in range(box)]
    return box

def crop_images_PURE_to_face_box(data_path, des_path):
    start = time.time()
    subjects = os.listdir(data_path)  # subject1,subject2
    subjects.sort()  #
    try:
        for subject in tqdm(subjects):
            # print(subject)
            # if subject != '03-06' and subject != '07-05':
            #     continue
            subject_path = os.path.join(data_path, subject)  #

            if os.path.isdir(subject_path):
                imgs_path = os.path.join(subject_path, subject)
                imgs = os.listdir(imgs_path)  #
                imgs.sort()
                des_sub_path = os.path.join(des_path, subject)  #
                flag = 0
                box = []
                for img in imgs:  # img.png
                    img_path = os.path.join(imgs_path, img)  # '/data1/vsign/PURE_filter_imgs/subject1/image_00000.jpg'
                    crop_face_path = os.path.join(des_sub_path,
                                                  img)  # '/data1/vsign/PURE_filter_crop_imgs/subject1/image_00000.jpg'

                    if not os.path.exists(des_sub_path):
                        os.makedirs(des_sub_path)

                    frame = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                    if flag == 0:
                        flag = 1
                        box = find_box(frame)
                    crop_img = frame[box[1]:box[3], box[0]:box[2]]
                    cv2.imwrite(crop_face_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        print(e, "\n")

    duration = str(datetime.timedelta(seconds=time.time() - start))
    print("It takes {} time for extracting PURE dataset".format(duration))


# img_path = 'D:\\DataSet\\RppgDataset\\PURE'
# deth_path = 'D:\\DataSet\\RppgDataset\\PURE_after\\PURE_filter_crop_imgs_box'
# crop_images_PURE_to_face_box(img_path,deth_path)

def crop_face_img(img_path,deth_path,crop_shape=(128,128)):
    print(img_path)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) # array
    # mtcnn = MTCNN(device='cuda:0')
    mtcnn = MTCNN(device='cpu:0')

    result= mtcnn.detect(img) # pytorch return  (boxes, probs) tensorflow return list[]
    h=img.shape[0] # 480
    w=img.shape[1] # 640

    # list [{'box': [110, 89, 235, 314],(x,y,width,height) 'confidence': 0.9999747276306152,
    # 'keypoints': {'left_eye': (181, 215), 'right_eye': (286, 218), 'nose': (235, 286), 'mouth_left': (190, 335), 'mouth_right': (273, 338)}}]

    # if len(result[0][0])==0:
    if result[0] is None:
        # If landmarks cannot be detected, return a bbx of first frame
        bounding_box= [0,0,w,h] # bounding_box [wmin,hmin,wmax,hmax],

    else:
        bounding_box = result[0][0] # bounding_box [wmin,hmin,wmax,hmax],[[286.418   151.44427 431.32965 348.65497]]

    # crop_img = img[max(0,int(bounding_box[1])):max(0,int(bounding_box[3])), max(0,int(bounding_box[0])):max(0,int(bounding_box[2]))] # (224,172,3) crop_img=img[hmin:hmax,wmin:wmax]
    crop_img = img[max(0, int(bounding_box[1]) - 60):max(0, int(bounding_box[3]) + 60),
               max(0, int(bounding_box[0]) - 60):max(0, int(
                   bounding_box[2]) + 60)]  # (224,172,3) crop_img=img[hmin:hmax,wmin:wmax]

    # resized_crop_img=cv2.resize(crop_img,crop_shape) #(128,128,3)

    # plt.figure()
    # plt.imshow(resized_crop_img)
    # plt.show()

    cv2.imwrite(deth_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

# img_path='test4.png'
# deth_path='ok4.png'
# crop_face_img(img_path,deth_path)

def crop_images_PURE_to_face(data_path,des_path):
    # img_path = '/data1/vsign/PURE'
    # deth_path = '/data1/vsign/PURE_crop_imgs'

    start = time.time()
    subjects = os.listdir(data_path) # subject1,subject2
    subjects.sort() #
    try:
        for subject in tqdm(subjects):
            print(subject)
            if subject != '03-06' and subject != '07-05':
                continue
            subject_path = os.path.join(data_path, subject) #

            if os.path.isdir(subject_path):
                imgs_path = os.path.join(subject_path, subject)
                imgs = os.listdir(imgs_path)  #
                imgs.sort()
                des_sub_path = os.path.join(des_path, subject) #

                for img in imgs:  # img.png
                    img_path = os.path.join(imgs_path, img) # '/data1/vsign/PURE_filter_imgs/subject1/image_00000.jpg'
                    crop_face_path = os.path.join(des_sub_path,img)  # '/data1/vsign/PURE_filter_crop_imgs/subject1/image_00000.jpg'

                    if not os.path.exists(des_sub_path):
                        os.makedirs(des_sub_path)
                    crop_face_img(img_path, crop_face_path)

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        print(e, "\n")

    duration = str(datetime.timedelta(seconds=time.time() - start))
    print("It takes {} time for extracting PURE dataset".format(duration))

# img_path = 'D:\\DataSet\\RppgDataset\\PURE'
# deth_path = 'D:\\DataSet\\homework'
# crop_images_PURE_to_face(img_path,deth_path)

# subjects=os.listdir('/data1/vsign/PURE_filter/PURE') #42, PURE_rPPG
# imgs_array=cv2.imread('/data1/vsign/PURE_filter_imgs/subject1/image_00000.jpg') #(480,640,3)





def PURE_img_to_numpy(root_path,des_path):

    if not os.path.exists(des_path):
        os.makedirs(des_path)

    start = time.time()
    subjects = os.listdir(root_path)  # 1,2,...10...
    subjects.sort()  #

    try:
        for subject in tqdm(subjects):  # 1,2
            # if subject != '03-06' and subject != '07-05':
            #     continue
            subject_path = os.path.join(root_path, subject)  # '/data1/vsign/VIPL_v2_crop_imgs/1'
            imgs = os.listdir(subject_path)  #
            imgs.sort()
            des_sub_path = os.path.join(des_path, subject)# '/data1/vsign/PURE_filter_crop_numpy/subject1'
            img_list=[]
            for img in imgs:
                img_path = os.path.join(subject_path, img)  # '/data1/vsign/PURE_filter_crop_imgs/subject1/image_00000.jpg'
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


root_path='D:\\DataSet\\RppgDataset\\PURE_after\\PURE_filter_crop_imgs_box'
des_path='D:\\DataSet\\RppgDataset\\PURE_after\\PURE_filter_crop_numpy_box'
PURE_img_to_numpy(root_path,des_path)



def convert_timestamp(current_timestamp, first_timestamp):
    return float(current_timestamp - first_timestamp) * 1e-9



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


def PURE_meta(root_path,des_path):
    # h5处理
    if not os.path.exists(des_path):
        os.makedirs(des_path)

    start = time.time()
    subjects = os.listdir(root_path)  # 1,2,...10...
    subjects.sort()  #

    try:
        for subject in tqdm(subjects):  # 1,2
            subject_path = os.path.join(root_path, subject)  #
            files = os.listdir(subject_path)  # video1, video2
            for file in files:
                if file.endswith('json'):
                    file_path = os.path.join(subject_path, file)  # '/data1/vsign/PURE_filter/PURE/subject1/ground_truth.txt'
                    # raw_data = []
                    with open(file_path, "r") as phys_file:
                        json_data = json.load(phys_file)
                        ids = ['/Image', '/FullPackage']
                        first_timestamp = min([min([entry['Timestamp'] for entry in json_data[id]]) for id in ids])
                        # 作下采样, 三次样条插值
                        # phys_data = {'ppg':[],'ppg_heart_rate':[] }
                        bvp = []
                        hr = []
                        for entry in json_data['/FullPackage']:
                            bvp.append(entry["Value"]["waveform"])
                            hr.append(entry["Value"]["pulseRate"])
                        T2 = len(bvp)
                        bvp_down = interpolate.CubicSpline(range(T2),bvp)
                        x_new = np.arange(0,T2,2)
                        hr_down = interpolate.CubicSpline(range(T2),hr)
                        phys_data = {'ppg': [bvp_down(x_new)], 'ppg_heart_rate': [hr_down(x_new)]}
                        df = pd.DataFrame(phys_data)
                        des_file_path = os.path.join(des_path, subject)
                        np.savez(des_file_path, hr=df['ppg_heart_rate'].values, wave=df['ppg'].values, fps_cal=30)
                        # 未作插值处理：
                        # phys_data = {'timestamp': [], 'ppg': [], 'ppg_heart_rate': [], 'o2sat': [], 'signal_quality': []}
                        # for entry in json_data['/FullPackage']:
                        #     phys_data['timestamp'].append(convert_timestamp(entry['Timestamp'], first_timestamp))
                        #     phys_data['ppg'].append(entry['Value']['waveform'])
                        #     phys_data['ppg_heart_rate'].append(entry['Value']['pulseRate'])
                        #     phys_data['o2sat'].append(entry['Value']['o2saturation'])
                        #     phys_data['signal_quality'].append(entry['Value']['signalStrength'] / 5.0)
                        # df = pd.DataFrame(phys_data)
                        # df = CorrectIrregularlySampledData(df, 30.0)
                        # des_file_path = os.path.join(des_path, subject)  # '/data1/vsign/PURE_filter_meta/subject1'
                        # df.to_excel(des_file_path + '.xlsx')
                        # np.savez(des_file_path, hr=df['ppg_heart_rate'].values, wave=df['ppg'].values,
                        #          frame_time=df['timestamp'].values, o2sat=df['o2sat'], signal_quality=df['signal_quality'], fps_cal=30)

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        print(e, "\n")

    duration = str(datetime.timedelta(seconds=time.time() - start))
    print("It takes {} time for extracting PURE dataset".format(duration))

# root_path='D:\\DataSet\\RppgDataset\\PURE'
# des_path='D:\\DataSet\\RppgDataset\\PURE_after\\PURE_filter_meta_numpy2'
# PURE_meta(root_path,des_path)


# phys_file = open("D:\\DataSet\\RppgDataset\\PURE\\01-01\\01-01.json", "r")
# json_data = json.load(phys_file)
# ids = ['/Image', '/FullPackage']
# first_timestamp = min([min([entry['Timestamp'] for entry in json_data[id]]) for id in ids])
# # 作下采样, 三次样条插值
# # phys_data = {'ppg':[],'ppg_heart_rate':[] }
# bvp = []
# hr = []
# for entry in json_data['/FullPackage']:
#     bvp.append(entry["Value"]["waveform"])
#     hr.append(entry["Value"]["pulseRate"])
# T2 = len(bvp)
# bvp_down = interpolate.CubicSpline(range(T2),bvp)
# x_new = np.arange(0,T2,2)
# hr_down = interpolate.CubicSpline(range(T2),hr)
# # print([bvp_down(x_new)])
# print(bvp_down(x_new))
# phys_data = {'ppg': [bvp_down(x_new)], 'ppg_heart_rate': [hr_down(x_new)]}
# df = pd.DataFrame(phys_data)