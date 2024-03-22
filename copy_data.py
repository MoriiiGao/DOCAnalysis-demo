import shutil
import os
from tqdm import tqdm

old_labels = '/data/gzc/VOCdata/PubLayNet_VOC/train/labels/'
old_images = '/data/gzc/VOCdata/PubLayNet_VOC/train/JPEGImages/'
new_tr_labels = '/data/gzc/yolo_publaynet/train/labels/'
new_tr_images = '/data/gzc/yolo_publaynet/train/images/'
new_va_labels = '/data/gzc/yolo_publaynet/val/labels/'
new_va_images = '/data/gzc/yolo_publaynet/val/images/'

label_list = os.listdir(old_labels)
for la in tqdm(label_list[0:7000]):
    filename = la.split('.')[0]
    label = filename + '.txt'
    image = filename + '.jpg'
    copy_la = old_labels + label
    copy_img = old_images + image

    if os.path.isfile(copy_la) and os.path.isfile(copy_img):
        shutil.copy(copy_la, new_tr_labels)
        shutil.copy(copy_img, new_tr_images)
    else:
        continue

for la in tqdm(label_list[8000:9000]):
    filename = la.split('.')[0]
    label = filename + '.txt'
    image = filename + '.jpg'
    copy_la = old_labels + label
    copy_img = old_images + image

    if os.path.isfile(copy_la) and os.path.isfile(copy_img):
        shutil.copy(copy_la, new_va_labels)
        shutil.copy(copy_img, new_va_images)
    else:
        continue

