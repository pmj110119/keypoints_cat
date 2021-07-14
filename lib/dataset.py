import os
import json
import math
import numpy as np

import torch
import torch.utils.data as data

import glob


from lib.utils import draw_umich_gaussian, gaussian_radius


import matplotlib.pyplot as plt

import albumentations as A
import cv2

transform = A.Compose([
    #A.RandomResizedCrop(2048,2048,scale=(0.5,1.0),ratio=(0.75,1.333),p=1.0),
    A.HorizontalFlip(p=0.5),   # 水平翻转
    #A.VerticalFlip(p=0.5),     # 垂直翻转
    # A.RandomRotate90(p=0.5),
    #A.IAAPiecewiseAffine(p=0.3),
    A.OneOf([
        A.RandomContrast( limit=0.1),
        A.RandomBrightness( limit=0.1),
        A.ColorJitter(brightness=0.1, contrast=0.1,
                saturation=0.1, hue=0.1),
        ]),
    ], keypoint_params=A.KeypointParams(format='xy')
)	


COCO_MEAN = np.array([0.8528331900647175, 0.6957574333240782, 0.6117719643342817],dtype=np.float32)
COCO_STD =  np.array([0.18984232639637347, 0.193275990180725, 0.17029545646390326],dtype=np.float32)

def loadLabel(file_path):
    with open(file_path, 'r') as f:
        line = f.read().split()
    return list(map(int,line))

class CAT_DATASET(data.Dataset):
    def __init__(self, data_dir, img_size=512,n_class=9,max_objs=18):
        super(CAT_DATASET, self).__init__()

        self.num_classes = n_class
        self.max_objs = max_objs
        self.images = []
        self.labels = []
        dirs = os.listdir(data_dir)
        for dir in dirs:
            self.images += glob.glob(os.path.join(data_dir,dir)+'/*.jpg')
        for img_path in self.images:
            self.labels.append(img_path+'.cat')
        self.num_samples = len(self.images)
        print('dataset samples = ', self.num_samples)

        self.down_ratio = 4
        self.img_size = img_size
        self.fmap_size = img_size//self.down_ratio


    def __getitem__(self, idx):

        # 读取图片
        img_path,label_path = self.images[idx],self.labels[idx]
        #print(img_path)
        #print(img_path,label_path)
        img = cv2.imread(img_path)
        label = loadLabel(label_path)[1:]
        new_label = []
        for i in range(self.num_classes):
            point = [float(label[i * 2])/ img.shape[1], float(label[i * 2 + 1])/img.shape[0]]
            new_label.append(point[0])
            new_label.append(point[1])

        img = cv2.resize(img,(self.img_size,self.img_size))


        img = (img/255.0-COCO_MEAN)/COCO_STD
        img = np.transpose(img,(2,0,1))     # h,w,c --> c,h,w
        img = torch.from_numpy(img.astype(np.float32, copy=False))
        new_label = torch.from_numpy(np.array(new_label).astype(np.float32))
        #print(new_label)
        return img, new_label



    def __len__(self):
        return self.num_samples




if __name__ == '__main__':


  dataset = CAT_DATASET('../cats')
  for i in range(10,20):
    img,label=dataset[i]
    img = cv2.imread('../cats/CAT_03/00000882_007.jpg')

    h,w,c = img.shape
    label = label.numpy()
    print(label)
    zz = []
    for i in range(9):
        points = [int(label[i * 2] * w), int(label[i * 2 + 1] * h)]
        zz += points
        print(points)
        cv2.circle(img, (points[0], points[1]), radius=10, thickness=-1, color=(0, 0, 255))
    cv2.imwrite('result.png', img)
    exit()
    pass


 