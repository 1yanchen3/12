import torch
import torch.nn as nn
from torchvision.transforms import functional as F
from torchvision import transforms as T
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
# 和模型有关
from basicseg.networks import build_network

class detector:
    def __init__(self, oup='ds', int8=False):
        # 网络位置
        self.opt = {'type': 'HCFnet2'}
        self.model = build_network(self.opt)
        # 训练权重
        modelWeightPath = ('/mnt/comic/zxr/cy/HCFNet-main1/experiment/AGPC/20241206_130603/models/net_best_mean.pth')
         #'/mnt/comic/zxr/cy/HCFNet-main1/experiment/AGPC/20241205_210602/models/net_best_mean.pth')
        # HCFnet1 /mnt/comic/zxr/cy/HCFNet-main1/experiment/AGPC/20241207_173234/models/net_best_mean.pth
        # DNANet/mnt/comic/zxr/cy/HCFNet-main1/experiment/AGPC/20241206_091626/models/net_best_mean.pth
        # res_UNet /mnt/comic/zxr/cy/HCFNet-main1/experiment/AGPC/20241205_125659/models/net_best_mean.pth
             #uiu /mnt/comic/zxr/cy/HCFNet-main1/experiment/AGPC/20241206_110317/models/net_best_mean.pth)
            #abc'/mnt/comic/zxr/cy/HCFNet-main1/experiment/AGPC/20241206_100918/models/net_best_mean.pth'
        # AGPCNet /mnt/comic/zxr/cy/HCFNet-main1/experiment/AGPC/20241205_210602/models/net_best_mean.pth

        # 5k agpc  /mnt/comic/zxr/cy/HCFNet-main1/experiment/HCF-5K/20241116_204334/models/net_best_mean.pth
        # 设备在cuda上，将模型放到cuda上
        self.device = 'cuda'
        self.model = self.model.to(self.device)
        # 这块是为了多个GPU并行计算忽略
        # self.model = nn.DataParallel(self.model)
        #  加载保存的模型权重
        self.model.load_state_dict(torch.load(modelWeightPath))
        # 将模型设置为评估模式，关闭dropout和batch normalization
        self.model.eval()
        # self.normTransform = T.Normalize(0.5, 0.25)
        # 将图像归一化到（0，1）
        self.normTransform = T.Normalize(0, 1)
        # 输出类型
        self.oup = oup
        self.int8 = int8

    def __call__(self, img):
        # h, w = img.shape
        img = self.pocessing(img).cuda()
        if self.oup == 'ds':
            output = self.model(img)
            output = self.normTransform(output)
            # output, _ = self.model(img)
            output = output.detach().cpu()
            resultMask = output[0][0]
            resultMask = torch.sigmoid(resultMask) * 255.
            resultMask = resultMask.numpy()
            if self.int8:
                resultMask = np.uint8(resultMask*255)
        elif self.oup == 'dsadd':
            a, b = self.model(img)
            a = a.detach().cpu().numpy()[0][0]
            b = b.detach().cpu().numpy()[0][0]
            x = np.array([a, b])
            output = np.max(x, axis=0)
            resultMask = output
            if self.int8:
                resultMask = np.uint8(resultMask*255)
        else:
            output = self.model(img)
            output = output.detach().cpu().numpy()
            resultMask = output[0][0]
            if self.int8:
                resultMask = np.uint8(resultMask*255)
        # resultMask = cv2.resize(resultMask, (w, h))
        # resultMask[resultMask >
        return resultMask

    def pocessing(self, img):
        # img = cv2.resize(img, (224, 224))
        # img = cv2.resize(img, (800, 608))

        # img = torch.tensor(img).to(self.device).unsqueeze(0).unsqueeze(0)
        # img = img / 255
        # img = self.normTransform(img)
        # return img

        img = F.to_pil_image(img)
        img = F.to_tensor(img)
        img = self.normTransform(img).unsqueeze(0)
        return img



if __name__ == '__main__':
    d = detector()
    input_dir = '/mnt/comic/zxr/cy/HCFNet-main1/datasets/SIRST/test/images/'
    output_dir = '/mnt/comic/zxr/cy/HCFNet-main1/predict/AUG/hcfnet2'

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # 检查文件扩展名
            path = os.path.join(input_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_COLOR)

            # 调整图片大小（如果需要）
            img = cv2.resize(img, (224, 224))
            out = d(img)
            #out = cv2.resize(out, (224, 224))

            # 保存预测结果
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, out)

            print(f'Saved prediction for {filename} to {output_path}')