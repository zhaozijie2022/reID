# my library for test
# Programed by Z.Zhao

# class TestConfig -- set the configuration for test
# def get_lc -- get labels and cameras from images list
# def load_test_data -- return DataLoaders(torch.utils.data.DataLoaders) and img_datasets(torchvision.dataset.ImageFolder)
# def load_model
# def extract_feature
# def save_res
# def hit_img
# def compute_ap_cmc
# def eval_test

# 排序: q=query > g=gallery ;  f=feature > l=label > c=cam
# fine_tune用了[10:17]8个人, test[1:9] 9个人


from __future__ import print_function, division
import torch
import numpy as np
from torch.autograd import Variable
import os
import cv2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import PCB, PCB_test
from tqdm import tqdm
import matplotlib.pyplot as plt


class TestConfig:
    """Set the configuration of test"""
    def __init__(self, is_tiny=True, is_RPP=False, is_det=True,
                 is_fine_tune=True, batch_size=12, which_epoch='last'):
        self.is_tiny = is_tiny  # use the sample of gallery
        self.is_RPP = is_RPP  # PCB or PCB+RPP
        self.is_det = is_det  # use the result of detection or gt
        self.is_fine_tune = is_fine_tune  # fine_tune_model?
        self.batch_size = batch_size
        self.which_epoch = which_epoch  # the last or the best
        self.result_path = './result/'  # the features save path
        self.feature_H = False  # 2048 or 256
        self.gpu_ids = [0]
        self.tiny = 'tiny' if is_tiny else 'std'
        self.det = 'det' if is_det else 'gt'
        self.stage = 'full' if is_RPP else 'PCB'
        self.model_path = './model_fine_tune/' if is_fine_tune else './model'
        abs_path = os.path.dirname(os.path.dirname(__file__))
        data_path = 'Dataset/UESTC_ReID/UESTC_ReID_' + self.det + '_' + self.tiny
        self.test_path = os.path.join(abs_path, data_path)  # test data path

    def print_info(self):
        print('-------------Test Options------------')
        print('The indices of GPUs: ' + str(self.gpu_ids))
        print('Detect/Ground Truth: ' + self.det)
        print('Tiny/Standard gallery: ' + self.tiny)
        print('Load the ' + self.which_epoch + ' epoch model')
        print('------------------------------------')


def get_lc(img_path):
    """get labels and cams from img_list"""
    labels, cameras = [], []
    for path, _ in img_path:
        filename = os.path.basename(path)
        label = filename[0:2]  # label
        camera = filename.split('cam')[1]  # camx
        labels.append(int(label))
        cameras.append(int(camera[0]))
    cameras = np.array(cameras)
    labels = np.array(labels)
    return labels, cameras


def load_test_data(test_opt):
    """Return dataloaders, img_datasets"""
    # 注意:如果is_fine_tune,应该读取query_fine_tune
    data_transforms = transforms.Compose([
        transforms.Resize((384, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 3个通道的均值、方差，此数据从ImageNet数据集计算出
    ])  # img pre-process
    data_path = test_opt.test_path
    if test_opt.is_fine_tune:
        query_path = os.path.join(data_path, 'query_ft')
    else:
        query_path = os.path.join(data_path, 'query')
    img_datasets = {'gallery': datasets.ImageFolder(os.path.join(data_path, 'gallery'), data_transforms),
                    'query':datasets.ImageFolder(query_path, data_transforms)}
    dataloaders = {x: DataLoader(img_datasets[x], batch_size=test_opt.batch_size, shuffle=False, num_workers=8)
                   for x in ['gallery', 'query']}
    return dataloaders, img_datasets


def load_model(test_opt):
    """Load the test model"""
    if len(test_opt.gpu_ids) > 0:
        torch.cuda.set_device(test_opt.gpu_ids[0])
    num_classes = 751  # the num of classes in Market-1501
    model = PCB(num_classes)
    if test_opt.is_RPP:
        model = model.convert_to_rpp()
    model_path = os.path.join(test_opt.model_path, test_opt.stage, 'net_%s.pth' % test_opt.which_epoch)
    model.load_state_dict(torch.load(model_path))
    model = PCB_test(model, featrue_H=test_opt.feature_H)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def extract_feature(model, dataloaders):
    """Extract features"""
    # feature存在cuda上, features存在cpu上(显存不够)
    features = torch.FloatTensor()
    for data in tqdm(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        inv_idx = torch.arange(w - 1, -1, -1).long()
        img_lr = img.index_select(3, inv_idx)  # 翻转之后的img

        output_1 = model(Variable(img.cuda()))
        feature = output_1.data.cpu()
        del output_1
        output_2 = model(Variable(img_lr.cuda()))
        feature += output_2.data.cpu()

        # feature size (n, 2048, 6) or (n, 256, 6)
        feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True) * np.sqrt(6)
        # 求tensor第dim=1维(2048 or 256)的p=2范数, 用于将特征归一化, sqrt(6)保证模=1 feature_norm.size (n, 1, 6)
        feature = feature.div(feature_norm.expand_as(feature))
        # 按照括号里的tensor的形状对fnorm展开(n, 1, 6)-->(n, 2048, 6), .div tensor除法, 即特征归一化
        feature = feature.view(feature.size(0), -1)  # 修改维度
        features = torch.cat((features, feature), 0)  # tensor拼接， 按维度0拼接
    return features


def save_res(test_opt, img_datasets, qf, gf):
    # feature, label, cam
    res = {'qf': qf.numpy(), 'gf': gf.numpy(), 'img_datasets':img_datasets}
    filename = 'features_' + test_opt.tiny + '_' + test_opt.stage + '_' + test_opt.det
    if test_opt.is_fine_tune:
        filename += '_ft'
    filename += '.npy'
    save_path = os.path.join(test_opt.result_path, filename)
    np.save(save_path, res)


def hit_img(path_q, img_datasets, gf, gl, gc, qf, ql, qc):
    # 传入1个qf, 在gallery中做选择, path_q为该图像的地址
    qf = qf.view(-1, 1)
    score = torch.mm(gf, qf)  # dot product
    score = score.squeeze(1).cpu()
    score = score.numpy()

    idx_sort = np.argsort(score)  # from small
    idx_sort = idx_sort[::-1]  # from large
    idx_same_c, idx_same_l = np.argwhere(qc == gc), np.argwhere(ql == gl)  # del the same cam
    # idx2elim = np.intersect1d(idx_same_l, idx_same_c)  # 在index中要被剔除的,  gallery中没有未分类(gl=-1)
    idx2elim = idx_same_c
    # np.intersect1d返回两个数组中相同的元素
    idx_sort = np.setdiff1d(idx_sort, idx2elim, assume_unique=True)
    # np.setdiff1d返回在arg1中但不在arg2中的值, 返回值从小到大排序且unique

    img = cv2.imread(path_q)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    fig = plt.figure('hit in gallery')
    ax = fig.add_subplot(3, 5, 3)
    ax.imshow(img)
    plt.title('q_%d_c%d' % (ql, qc))
    plt.axis('off')

    for i in range(10):
        idx = idx_sort[i]
        tar_path, _ = img_datasets['gallery'].imgs[idx]
        img = cv2.imread(tar_path)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        ax = fig.add_subplot(3, 5, i + 6)
        ax.imshow(img)
        plt.title('g_%d_c%d' % (gl[idx], gc[idx]))
        plt.axis('off')
    plt.show()


def compute_ap_cmc(qf, ql, qc, gf, gl, gc):
    ap = 0
    cmc = torch.IntTensor(len(gl)).zero_()

    qf = qf.view(-1, 1)
    score = torch.mm(gf, qf)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    idx_sort = np.argsort(score)  # from small to large
    idx_sort = idx_sort[::-1]

    idx_same_c, idx_same_l = np.argwhere(gc == qc), np.argwhere(gl == ql)
    idx_P = np.setdiff1d(idx_same_l, idx_same_c, assume_unique=True)  # 实际的正例, 相同label不同cam
    # idx2elim = np.intersect1d(idx_same_l, idx_same_c)
    idx2elim = idx_same_c
    idx_sort = np.setdiff1d(idx_sort, idx2elim, assume_unique=True)  # 剔除相同cam


    num_P = len(idx_P)  # 实际的正例个数
    mask = np.in1d(idx_sort, idx_P)
    # np.in1d判断arg1是否在arg2中, 返回与arg1同型的bool数组,
    where_right = np.argwhere(mask == True)  # 返回非0元素的索引, 即按照返回mask中True对应元素的索引
    where_right = where_right.flatten()
    # where_right储存的是对idx_sort的索引, 指idx_sort中的第几个元素是idx_P中的, 即 idx_sort中预测正确的元素的位置

    cmc[where_right[0]:] = 1
    for i in range(num_P):  # 循环来遍历PR曲线上的每个点, 积分产生ap
        d_recall = 1.0 / num_P
        #  由于idx_sort包含idx_P, 所以所有的正例必被检测出来, 只是排序不同, 所以d_recall必增加1
        precision = (i + 1) * 1.0 / (where_right[i] + 1)
        #  当前有(i+1)个被检测出来, 现在一共给出了 (where_right[i]+1)个预测结果
        if where_right[i] != 0:
            old_precision = i * 1.0 / where_right[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2
    # 使用梯形积分方法
    if cmc[9] == 0:
        print('First 10 fail in q_%d_c%d' % (ql, qc))
    return ap, cmc


def eval_test(features_path):
    # print(os.path.basename(features_path))
    result = np.load(features_path, allow_pickle=True).item()
    qf, gf = torch.FloatTensor(result['qf']), torch.FloatTensor(result['gf'])
    qf, gf = qf.cuda(), gf.cuda()
    img_datasets = result['img_datasets']

    gl, gc = get_lc(img_datasets['gallery'].imgs)
    ql, qc = get_lc(img_datasets['query'].imgs)

    num_q, num_g = len(ql), len(gl)
    cmc = torch.IntTensor(num_g).zero_()
    ap = 0.0
    for i in range(num_q):
        ap_tmp, cmc_tmp = compute_ap_cmc(qf=qf[i], ql=ql[i], qc=qc[i], gf=gf, gl=gl, gc=gc)
        cmc += cmc_tmp
        ap += ap_tmp
    cmc = cmc.float()
    cmc = cmc / num_q
    plt.plot(cmc[0:10])
    plt.show()
    mAP = ap / num_q
    print('Rank-1:%f Rank-5:%f Rank-10:%f mAP:%f' % (cmc[0], cmc[4], cmc[9], mAP))
    print('First-10: %d/%d'%(int(num_q*cmc[9]), num_q))



