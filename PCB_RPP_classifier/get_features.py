# 对gallery和query提取特征, 并使用npy格式存储
# Programed by Z.Zhao

from __future__ import print_function, division
from mylib4test import *

is_tiny = False  # gallery_tiny or gallery_std
is_RPP = True  # PCB or PCB_RPP
is_det = False  # gallery_det or gallery_gt
is_fine_tune = False
batch_size = 12
which_epoch = 'last'


if __name__ == "__main__":
    test_opt = TestConfig(is_tiny=is_tiny, is_RPP=is_RPP, is_det=is_det,
                            batch_size=batch_size, is_fine_tune=is_fine_tune, which_epoch=which_epoch)
    test_opt.print_info()

    dataloaders, img_datasets = load_test_data(test_opt)
    gl, gc = get_lc(img_datasets['gallery'].imgs)
    ql, qc = get_lc(img_datasets['query'].imgs)
    print('Load data, done!')

    model = load_model(test_opt)
    print('Load model, done!')

    print("Extract features in query, %d images" % len(ql))
    qf = extract_feature(model, dataloaders['query'])

    print("Extract features in gallery, %d images" % len(gl))
    gf = extract_feature(model, dataloaders['gallery'])
    print("Extract features, done!")

    save_res(test_opt, img_datasets, qf=qf, gf=gf)
    print("Save features, done!")



