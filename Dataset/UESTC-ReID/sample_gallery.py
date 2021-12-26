# sample gallery, get num_img/ID/cam
# Programed by Z.Zhao

import os
import shutil
import random


num_img = 20
source_path = "./UESTC_ReID_det_std/gallery/"
target_path = "./UESTC_ReID_det_tiny/gallery/"

# source_path = "./UESTC_ReID_gt_std/gallery/"
# target_path = "./UESTC_ReID_gt_tiny/gallery/"


def set_dir(target_path):
    '''初始化目标地址'''
    ids = ["%02d" % i for i in range(1, 18)]
    for id in ids:
        file_path = target_path + id + '/'
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        else:
            shutil.rmtree(file_path)
            os.mkdir(file_path)


def dict_init():
    '''初始化字典'''
    cams = ["cam%d" % i for i in range(1, 7)]
    ids = ["%02d" % i for i in range(1, 18)]
    std_id_cam_dict = {}  # key为id, value为字典, value字典的key为cam, value为该cam下的frame
    tiny_cam_dict = {}
    for id in ids:
        std_id_cam_dict[id] = {}
        tiny_cam_dict[id] = {}
        for cam in cams:
            std_id_cam_dict[id][cam] = []
            tiny_cam_dict[id][cam] = []
    std_id_cam_dict['16'].pop('cam3')
    tiny_cam_dict['16'].pop('cam3')
    return std_id_cam_dict, tiny_cam_dict


def tiny_sample(source_path, num_img, std_dict, tiny_dict):
    '''读取src, '''
    ids = ["%02d" % i for i in range(1, 18)]
    #  读取std
    for id in ids:
        id_path = os.path.join(source_path, id)
        img_list = os.listdir(id_path)
        for img in img_list:
            img = img.strip('.jpg')
            tmp_id, tmp_cam, tmp_frame = img.split('_')
            std_dict[tmp_id][tmp_cam].append(tmp_frame)
    # 采样
    for id in std_dict.keys():
        for cam in std_dict[id].keys():
            if len(std_dict[id][cam]) >= num_img:
                tiny_dict[id][cam] = random.sample(std_dict[id][cam], num_img)
            else:
                tiny_dict[id][cam] = std_dict[id][cam]
    return tiny_dict


if __name__ == "__main__":
    cams = ["cam%d" % i for i in range(1, 7)]
    ids = ["%02d" % i for i in range(1, 18)]

    set_dir(target_path)
    std_dict, tiny_dict = dict_init()
    tiny_dict = tiny_sample(source_path, num_img, std_dict, tiny_dict)

    for id in tiny_dict.keys():
        for cam in tiny_dict[id].keys():
            for img in tiny_dict[id][cam]:
                img_name = id + '_' + cam + '_' + img + '.jpg'
                img_src = os.path.join(source_path, id, img_name)
                img_tar = os.path.join(target_path, id, img_name)
                shutil.copyfile(img_src, img_tar)














