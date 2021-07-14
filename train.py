# python imports
import os
import glob
import warnings
# external imports
import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
# model imports

from lib.network import CatNet

from lib.dataset import CAT_DATASET
# utils imports
from lib.utils import _tranpose_and_gather_feature
from lib.losses import _neg_loss, _reg_loss
import torch.nn as nn
import time
import torch
#torch.cuda.set_device(1)




def train():
    save_iters = 100    # 每迭代100次保存一次模型
    load_iters = 4000   # 加载之前训练的模型(指定迭代次数)

    model = CatNet()    # 实例化模型

    # 加载权重
    if(load_iters!=0):
        pre=torch.load(os.path.join('./Checkpoint',str(load_iters)+'.pth'))
        model.load_state_dict(pre)

    model.train()
    model = model.cuda()
    opt = Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    # 加载训练数据
    dataset = CAT_DATASET("./cats")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=False)

    # Training loop.
    iters = load_iters + 1
    while iters < 10000:
        time_start = time.time()
        for img,label in dataloader:
            iters = iters+1

            # 数据存入gpu
            img = img.cuda()
            label = label.cuda()

            # 模型预测
            output = model(img)

            # 计算loss
            loss = loss_fn(output, label)

            # 反向传播
            opt.zero_grad()
            loss.backward()
            opt.step()

            # 保存模型
            if(iters% save_iters == 0):
                save_file_name = os.path.join('Checkpoint', '%d.pth' % iters)
                torch.save(model.state_dict(), save_file_name)
            print('iters{%d} ' % (iters) +' loss= %.5f   time: %.1f' %(loss.item(), time.time()-time_start) )


if __name__ == "__main__":
    train()
