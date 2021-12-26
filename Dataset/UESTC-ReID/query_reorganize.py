
import os
import shutil

src_path = "./query/"
tar_path = './query_test/'

cams = ["cam" + '%d' % i for i in range(1, 7)]
ids = ["%02d" % i for i in range(1, 18)]

if os.path.exists(tar_path):
    shutil.rmtree(tar_path)
os.mkdir(tar_path)

for id in ids:
    os.mkdir(os.path.join(tar_path, id))

for cam in cams:
    img_path = os.path.join(src_path, cam)
    for img in os.listdir(img_path):
        img_tmp = img.strip('.jpg')
        idx_cam, id = map(int, img_tmp.split('_'))
        img_src = os.path.join(img_path, img)
        img_tar = os.path.join(tar_path, '%02d/%02d_cam%s_%s.jpg' %(id, id, idx_cam, idx_cam))
        shutil.copyfile(img_src, img_tar)
