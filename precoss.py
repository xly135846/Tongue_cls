import os
import cv2
import glob
import random
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

random.seed(2021)

os.environ["CUDA_VISIBLE_DEVICES"]="4"
NUM_CLASS = 4

figure_img_files = glob.glob("D:/jupyter/shouzhi_seg/figure/img_*")
figure_mask_files = glob.glob("D:/jupyter/shouzhi_seg/figure/mask_*")
figure_img_files.sort()
figure_mask_files.sort()
figure_img_array = []
figure_mask_array = []
for i in range(len(figure_img_files)):
    img = cv2.imread(figure_img_files[i],1)
    mask = cv2.imread(figure_mask_files[i],0)
    figure_img_array.append(img)
    figure_mask_array.append(mask)
print("zd_file",len(figure_img_array),len(figure_mask_array))

background_files = glob.glob("D:/jupyter/background/*")
background_files.sort()
background_array = []
for i in range(len(background_files)):
    tmp = cv2.imread(background_files[i],1)
    tmp = cv2.resize(tmp,(128,128))
    background_array.append(tmp)
print("background_file",len(background_array))

def motion_blur(image, degree=12, angle=45):
    image = np.array(image)
 
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
 
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
 
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

def get_list_from_filenames(file_path):
    with open(file_path,'r',) as f:
        lines = [one_line.strip('\n') for one_line in f.readlines()]
    return lines

def imgBrightness(img, label, random_degree):
    rnd = np.random.random_sample()

    if rnd < random_degree:
        line_degree = random.randint(15,85)/100
        blank = np.zeros([100, 100, 1], img.dtype)
        img = cv2.addWeighted(img, line_degree, blank, 1-line_degree, 1)
    return img, label

def enforce_random(img, label, random_degree):

    rnd = np.random.random_sample()
    if rnd <random_degree:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if label == 1:
            label = 3
        elif label == 3:
            label = 1
        else:
            label = label

    rnd = np.random.random_sample()
    if rnd < random_degree:
        img = img.filter(ImageFilter.BLUR)

    if True:
        ds = 1 + np.random.randint(0,2)
        original_size = img.size
        img = img.resize((int(img.size[0] / ds), int(img.size[1] / ds)), resample=Image.BILINEAR)
        img = img.resize((original_size[0], original_size[1]), resample=Image.BILINEAR)

    return img, label

def seg_enforce_random(img, label,random_degree):
    
    rnd = np.random.random_sample()
    if rnd < random_degree:
        img = cv2.flip(img,1)
        label = cv2.flip(label,1)

    return img, label

def motion_random(img, label, random_degree):
    rnd = np.random.random_sample()

    if rnd < random_degree:
        ### origin
        random_degree = random.randint(12,20)
        # random_degree = random.randint(10,25)
        img = motion_blur(img, degree=random_degree, angle=45)
    
    return img, label

def shouzhi_zd_random(img, label, random_degree):
    rnd = np.random.random_sample()
    if rnd < random_degree:
        index = random.randint(0,len(figure_img_files)-1)
        zd_array = figure_img_array[index]
        zd_mask_array = figure_mask_array[index]
        img[zd_array!=0]=zd_array[zd_array!=0]
        label[zd_mask_array==200]=0
    return img,label

def rec_zd_random(img, label, random_degree):
    rnd = np.random.random_sample()
    if rnd < random_degree:
        index = random.randint(0,len(background_array)-1)
        zd_array = background_array[index]

        rec_num = random.randint(2,6)
        aaa,bbb = [],[]
        for i in range(2):
            aaa.append(random.randint(0,img.shape[2]))
        for i in range(rec_num*2):
            bbb.append(random.randint(0,img.shape[1]))
        points = aaa+bbb+aaa
        points = np.array(points).reshape(-1,2)
        
        background_up = np.zeros(img.shape)
        background_up = cv2.fillPoly(background_up, np.int32([points]),(200,200,200))
        background_up[background_up!=200]=1
        background_up[background_up==200]=0
        img_tmp = background_up*img
        mask_tmp = background_up[:,:,0]*label
        
        background_up[background_up==0]=2
        background_up[background_up==1]=0
        background_up[background_up==2]=1
        background_tmp = background_up*zd_array
    
        result = img_tmp + background_tmp
        return result, mask_tmp 
    else:
        return img, label

def seg_convert_mask_line(aaa,mode):
    tmp_list = [1,2]
    fu_fprs21c_list = [0, 11, 12, 255]
    sjt_fprs21x_list = [0, 225, 76, 255]
    if "fu_fprs21c" in mode:
        aaa[aaa==1]=0
        aaa[aaa==11]=1
        aaa[aaa==12]=1
        aaa[aaa==255]=1
        for i in fu_fprs21c_list:
            if i not in tmp_list:
                aaa[aaa==i]=0
    
    elif ("sjt_fprs21x" in mode) or ("mouth_bezier_land2seg" in mode) or ("mouth_land2seg" in mode):
        aaa[aaa==226]=1
        aaa[aaa==76]=1
        aaa[aaa==255]=1
        for i in sjt_fprs21x_list:
            if i not in tmp_list:
                aaa[aaa==i]=0
                   
    else:
        print("error")
    
    return aaa

def seg_convert(aaa,mode):
    tmp_list = [0,1]
    if "fu_fprs21c" in mode:
        aaa[aaa==1]=0
        aaa[aaa==11]=1
        aaa[aaa==12]=1
        for i in range(256):
            if i not in tmp_list:
                aaa[aaa==i]=0
    
    elif ("sjt_fprs21x" in mode) or ("mouth_bezier_land2seg" in mode) or ("mouth_land2seg" in mode):
        aaa[aaa==226]=1
        aaa[aaa==76]=1
        for i in range(256):
            if i not in tmp_list:
                aaa[aaa==i]=0
                   
    else:
        print("error")
    
    return aaa

def box_random(img, label):

    return img, label

if __name__=="__main__":

    word2number_dict = {
        "0":0,
        "0_new":0,
        "0_new1":0,
        "0_new2":0,
        "0_new3": 0,
        "0_new4":0,
        "neg":0,
        "gen_minzui":0,
        "shoushi_extra": 0,
        "20220110_imgs_npy_0": 0,
        "ziya_kouhong_shoushi":0,
        "shoushi_quanzhedang":0,
        "1": 1,
        "1_train": 1,
        "2":2,
        "2_new":2,
        "2_new_cp":2,
        "2_new_yisifuyangben": 2,
        "2_yisifuyangben": 2,
        "20220110_imgs_npy_2": 2,
        "ziya_shenshetou_2":2,
        "3":3,
        "3_train": 3,
    }

    IDs = get_list_from_filenames('f:/data_segment/0318_IDs.txt')
    print(len(IDs))
    
    # for i in range(len(IDs[:])):
    #     tmp = IDs[i].replace("_0223","_0318")
    #     if os.path.exists(tmp)==False:
    #         print(IDs[i])
    
    train_IDs = random.sample(IDs, int(len(IDs)*0.95))
    val_IDs = list(set(IDs)-set(train_IDs))
    
    print(len(train_IDs),len(val_IDs))
    
    for i in tqdm(range(len(train_IDs))):
        f = open('f:/data_segment/0318_train_IDs.txt','a')
        f.write('\n'+str(train_IDs[i]))
        f.close()    
    
    for i in tqdm(range(len(val_IDs))):
        f = open('f:/data_segment/0318_val_IDs.txt','a')
        f.write('\n'+str(val_IDs[i]))
        f.close()    
    
    train_list,val_list = [],[]
    all_jpg_names = get_list_from_filenames('f:/data_segment/0318_segment_files.txt')
    for i in tqdm(range(len(all_jpg_names))):
        tmp = all_jpg_names[i].split("/")[5].replace(all_jpg_names[i].split("/")[5].split("_")[-1], "")
        if tmp in train_IDs:
            train_list.append(all_jpg_names[i])
        if tmp in val_IDs:
            val_list.append(all_jpg_names[i])

    print(len(train_list),len(val_list))

    for i in tqdm(range(len(train_list))):
        f = open('f:/data_segment/0318_train_list.txt','a')
        f.write('\n'+str(train_list[i]))
        f.close() 
    
    for i in tqdm(range(len(val_list))):
        f = open('f:/data_segment/0318_val_list.txt','a')
        f.write('\n'+str(val_list[i]))
        f.close()

    # txt_list = [
    #             # "f:/data_segment/background_5/0316_train_list0.txt",
    #             # "f:/data_segment/background_5/0316_train_list1.txt",
    #             # "f:/data_segment/background_5/0316_train_list2.txt",
    #             "f:/data_segment/background_5/0316_train_list4.txt",]

    # train_list = []
    # for i in txt_list:
    #     images_list = get_list_from_filenames(i)
    #     train_list += images_list
    # print(len(train_list))    
    
    # ccc = get_list_from_filenames("f:/data_segment/segment_files.txt")
    # print(len(ccc))
    # aaa = [i.split("/")[5].replace(i.split("/")[5].split("_")[-1], "") for i in ccc]
    # bbb = list(set(aaa))
    # print(len(bbb))
    # print(bbb[:10])
    

    # tmp = [i.split("/")[4] for i in ccc]
    # print(list(set(tmp)))
    
    # # tmp = [i.split("/")[4] for i in images_list]
    # print(len(image_list))
    
    # train_list = []
    # all_label = []
    # for i in range(len(images_list)):
    #     label = word2number_dict[images_list[i].split("/")[4]]
    #     if label==0:
    #         if int(images_list[i][-5])<4:
    #             all_label.append(label)
    #             train_list.append(images_list[i])
    #     else:
    #         all_label.append(label)
    #         train_list.append(images_list[i])
    # print(len(train_list))

    # train_list = []
    # all_label = []
    # for i in range(len(images_list)):
    #     label = word2number_dict[images_list[i].split("/")[4]]
    #     if :
    #         train_list.append(images_list[i])
    # print(len(train_list))

    # for i in tqdm(range(len(tmp))):
    #     f = open('f:/data/tmp.txt','a')
    #     f.write('\n'+str(tmp[i]))
    #     f.close()
    
    train_list = get_list_from_filenames("f:/data_segment/0318_train_list.txt")
    print(len(train_list))
    random.shuffle(train_list)
    num = int(len(train_list)/5)+1
    print(num)
    for kkk in range(5):
        for i in tqdm(range(len(train_list[kkk*num:(kkk+1)*num]))):
            f = open('f:/data_segment/data_0321/031_train_list'+str(kkk)+'.txt','a')
            f.write('\n'+str(train_list[i+kkk*num]))
            f.close()
    # all_label = []
    # count = [[],[],[],[]]
    # for i in tqdm(range(len(train_list))):
    #     label = word2number_dict[train_list[i].split("/")[4]]
    #     for i in range(4):
    #         if int(label)==i:
    #             count[i].append(label)
    #     all_label.append(all_label)    
    # for i in range(4):
    #     print(i,len(count[i]))
    
    # print(len(images_list),len(aaa))
    # for i in tqdm(range(len(aaa[:]))):
    #     f = open('f:/data/aaa.txt','a')
    #     f.write('\n'+str(aaa[i]))
    #     f.close()
    
    
    
    ### label==1 and label==3 
    # files = glob.glob("G:/data/tongue/20220211/3_train/*.png")

    # for i in tqdm(range(len(files[:]))):
    #     f = open('f:/data/1_trian_ran.txt','a')
    #     f.write('\n'+str(files[i]))
    #     f.close()

    ### segmention
    # files_fu_fprs21c = glob.glob("G:/data/segmentation/fu_fprs21c_0318/*/*.png")
    # print(len(files_fu_fprs21c))
    # files_sjt_fprs21x = glob.glob("G:/data/segmentation/sjt_fprs21x_0318/*/*.png")
    # print(len(files_sjt_fprs21x))
    # files_fu_fprs21c.sort()
    # files_sjt_fprs21x.sort()
    # files = files_sjt_fprs21x+files_fu_fprs21c
    # print(len(files))
    
    # for i in tqdm(range(len(files[:]))):
    #     f = open('f:/data_segment/segment_files_0318.txt','a')
    #     f.write('\n'+str(files[i]))
    #     f.close()
    # files = glob.glob("G:/data/segmentation/quanzhedang/*/*.png")
    
    # for i in tqdm(range(len(files))):
    #     f = open('f:/data_segment/quanzhedang.txt','a')
    #     f.write('\n'+str(files[i]))
    #     f.close() 
    
    
    # all_files = get_list_from_filenames("f:/data_segment/0318_segment_files.txt")
    
    # aaa = [i.split("/")[5].replace(i.split("/")[5].split("_")[-1], "") for i in all_files]
    # bbb = list(set(aaa))
    # print(len(bbb))

    # for i in tqdm(range(len(bbb[:]))):
    #     f = open('f:/data_segment/0318_IDs.txt','a')
    #     f.write('\n'+str(bbb[i]))
    #     f.close()