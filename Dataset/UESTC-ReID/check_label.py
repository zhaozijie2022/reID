# 检查label中的bbox, 剔除错误项
# Programed by Z.Zhao

import cv2
import os
import shutil
from tqdm import tqdm


if __name__ == '__main__':
    label_path = './label/'
    save_path = './label_check/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        shutil.rmtree(save_path)
        os.mkdir(save_path)

    cams = ["cam" + '%d' % i for i in range(1, 7)]

    for cam in cams:
        print("Current:", cam)
        label_path_cam = label_path + cam + ".txt"

        with open(label_path_cam, 'r') as label_cam:
            label_data = label_cam.readlines()
            # frame_idx, id, cen_x, cen_y, w, h
            # 根据label_data截取图像并按照id存储
            for line_label in label_data:
                line_label = line_label.strip('\n')  # 去除读取出的'\n'
                frame_idx, id, cen_x, cen_y, w, h = list(map(int, line_label.split(',')))  # 分割并转化为数字列表
                line_label += '\n'
                if cen_y - h // 2 < 0 or cen_x - w // 2 < 0 or cen_y + h // 2 >= 1080 or cen_x + w // 2 >= 1920:
                    print(cam + '-error: ' + line_label)
                    label_data.remove(line_label)
        with open(save_path + cam + ".txt", 'w') as label_cam:
            label_cam.writelines(label_data)
