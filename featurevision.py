import os
import torch
import torch.nn as nn
import tensorboardX as tbX
import argparse
import cv2
import imageio
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
from scipy import misc
from models.experimental import attempt_load

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")

#模型加载
weights = './runs/train/exp2/weights/best.pt'
model = attempt_load(weights)

# 数据加载及预处理
img_path = "./data/PMC1180429_00002.jpg" # your path to image
img = Image.open(img_path)

writer = tbX.SummaryWriter('./runs/features')

image = cv2.imread(img_path, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)

image_process = transforms.Compose([
    transforms.Resize(size=(640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 归一化图像，可选
])

image = image_process(image).unsqueeze(dim=0)  # 1,c,h,w
print(image.shape)
image = image.to(device)

model.eval()
with torch.no_grad():
    pred, extract_result = model(image)
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    pred = pred.squeeze().cpu().numpy()

    for i in range(len(extract_result)):
        x = torch.squeeze(extract_result[i])
        x = x.permute(1, 0, 2, 3).cpu()  # 1,c,h,w--->c,1,h,w
        x = make_grid(x)
        # print(extract_result[i].shape)
        writer.add_image('step' + str(i), x)

    path = 'pred_img'
    if not os.path.exists(path):
        os.makedirs(path)
    imageio.imsave('pred_img', pred)

