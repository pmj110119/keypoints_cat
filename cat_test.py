import os,glob
import torch
import numpy as np
from lib.network import CatNet
import cv2
device = torch.device('cuda:0')


COCO_MEAN = np.array([0.8528331900647175, 0.6957574333240782, 0.6117719643342817],dtype=np.float32)
COCO_STD =  np.array([0.18984232639637347, 0.193275990180725, 0.17029545646390326],dtype=np.float32)

# 待预测图片路径和保存路径
root = 'test'
target_folder = root+'/CAT_06'
saved_folder = root+'/result'
if not os.path.exists(root):
    os.makedirs(root)
if not os.path.exists(saved_folder):
    os.makedirs(saved_folder)

def test():

    pth_epoch = 4000 # 加载之前训练的模型(指定轮数)

    # 实例化网络
    model = CatNet().to(device)
    # 加载训练好的权重
    model.load_state_dict(torch.load(os.path.join('./Checkpoint',str(pth_epoch)+'.pth')))
    model.eval()

    # 一张一张处理指定文件夹内的图像
    imgs = glob.glob(target_folder+'/*.jpg')
    for img_path in imgs:
        print(img_path)

        # 加载图像
        src = cv2.imread(img_path)
        h, w, c = src.shape

        # 不进行反向传播
        with torch.no_grad():
            # 数据预处理
            img = cv2.resize(src, (512,512))
            img = (img / 255.0 - COCO_MEAN) / COCO_STD
            img = np.transpose(img, (2, 0, 1))  # h,w,c --> c,h,w
            img = torch.from_numpy(img.astype(np.float32, copy=False)).unsqueeze(0).to(device)

            # 模型预测
            output = model(img).cpu().numpy()[0]

            # 结果可视化
            zz = []
            for i in range(9):
                points = [int(output[i*2]*w),int(output[i*2+1]*h)]
                zz += points
                cv2.circle(src,(points[0],points[1]),radius=10,thickness=-1,color=(0,0,255))
            cv2.imwrite(os.path.join(root,'result',os.path.basename(img_path)),src)



if __name__ == "__main__":
    test()
